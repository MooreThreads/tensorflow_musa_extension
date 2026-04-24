/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mu/graph_fusion/clip_fusion.h"

#include <algorithm>
#include <set>

#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  std::string node_name = FusionGraphUtils::GetProducerNodeName(input);
  if (node_name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, node_name);
}

int CountConsumersExcluding(const GraphDef& graph, const std::string& node_name,
                            const std::set<std::string>& excluded_consumers) {
  int count = 0;
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (excluded_consumers.count(consumer.name()) > 0) {
      continue;
    }
    for (int j = 0; j < consumer.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(j)) == node_name) {
        ++count;
      }
    }
  }
  return count;
}

}  // namespace

MusaClipFusion::MusaClipFusion() = default;

bool MusaClipFusion::IsKernelAvailable() const {
  if (IsDisabledByEnv()) {
    return false;
  }
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaClipFusion::Match(const GraphDef& graph,
                                        int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);
  if (IsOp(start_node, "Maximum")) {
    return MatchFromMaximumNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaClipFusion::MatchFromMaximumNode(
    const GraphDef& graph, int maximum_node_idx) const {
  FusionMatchResult result;
  const NodeDef& maximum_node = graph.node(maximum_node_idx);

  if (!IsOp(maximum_node, "Maximum") || maximum_node.input_size() != 2) {
    return result;
  }

  int minimum_input_idx = -1;
  const NodeDef* minimum_node = FindProducer(graph, maximum_node.input(0));
  if (minimum_node && IsOp(*minimum_node, "Minimum")) {
    minimum_input_idx = 0;
  } else {
    minimum_node = FindProducer(graph, maximum_node.input(1));
    if (minimum_node && IsOp(*minimum_node, "Minimum")) {
      minimum_input_idx = 1;
    }
  }

  if (minimum_input_idx < 0 || !minimum_node || minimum_node->input_size() != 2) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&maximum_node, minimum_node};
  result.captured_nodes["inner"] = minimum_node;
  result.captured_nodes["output"] = &maximum_node;
  result.captured_attrs["x_input"] = minimum_node->input(0);
  result.captured_attrs["hi_input"] = minimum_node->input(1);
  result.captured_attrs["lo_input"] = maximum_node.input(1 - minimum_input_idx);
  return result;
}

Status MusaClipFusion::Apply(GraphDef* graph,
                             const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid Clip match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto inner_it = match_result.captured_nodes.find("inner");
  auto x_input_it = match_result.captured_attrs.find("x_input");
  auto lo_input_it = match_result.captured_attrs.find("lo_input");
  auto hi_input_it = match_result.captured_attrs.find("hi_input");
  if (output_it == match_result.captured_nodes.end() ||
      inner_it == match_result.captured_nodes.end() ||
      x_input_it == match_result.captured_attrs.end() ||
      lo_input_it == match_result.captured_attrs.end() ||
      hi_input_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured nodes for MusaClip fusion");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* inner_node = inner_it->second;
  if (!output_node || !inner_node) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output/inner nodes for MusaClip fusion");
  }

  const std::string original_name = output_node->name();
  const std::string original_device = output_node->device();
  const std::string inner_node_name = inner_node->name();
  const std::string x_input = x_input_it->second;
  const std::string lo_input = lo_input_it->second;
  const std::string hi_input = hi_input_it->second;
  AttrValue output_type_attr;
  bool has_output_type = false;
  auto t_it = output_node->attr().find("T");
  if (t_it != output_node->attr().end()) {
    output_type_attr = t_it->second;
    has_output_type = true;
  }

  const_cast<NodeDef*>(output_node)->set_name(original_name + "_original");
  const std::string renamed_output_name = output_node->name();

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaClip");
  fused_node->set_device(original_device);
  fused_node->add_input(x_input);
  fused_node->add_input(lo_input);
  fused_node->add_input(hi_input);

  auto* attr = fused_node->mutable_attr();
  if (has_output_type) {
    (*attr)["T"] = output_type_attr;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }

  std::set<std::string> nodes_to_remove;
  nodes_to_remove.insert(renamed_output_name);

  if (CountConsumersExcluding(*graph, inner_node_name, nodes_to_remove) == 0) {
    nodes_to_remove.insert(inner_node_name);
  }

  std::vector<int> indices_to_remove;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_remove.count(graph->node(i).name()) > 0) {
      indices_to_remove.push_back(i);
    }
  }

  std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
  for (int idx : indices_to_remove) {
    FusionGraphUtils::RemoveNode(graph, idx);
  }

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaClipFusion);
REGISTER_FUSION_KERNEL(MusaClipFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
