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

#include "mu/graph_fusion/safe_clip_fusion.h"

#include <algorithm>
#include <set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/errors.h"

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

}  // namespace

MusaSafeClipFusion::MusaSafeClipFusion() = default;

bool MusaSafeClipFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

// Pattern: Select(IsNan(x), Const(0), Maximum(Minimum(x, hi), lo))
FusionMatchResult MusaSafeClipFusion::Match(const GraphDef& graph,
                                            int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& select_node = graph.node(start_node_idx);
  if ((!IsOp(select_node, "Select") && !IsOp(select_node, "SelectV2")) || select_node.input_size() != 3) {
    return result;
  }

  // 1. Condition: IsNan
  const NodeDef* isnan_node = FindProducer(graph, select_node.input(0));
  if (!isnan_node || !IsOp(*isnan_node, "IsNan")) {
    return result;
  }

  // 2. then_value: Fill(0) or Const(0)
  const NodeDef* zero_node = FindProducer(graph, select_node.input(1));
  if (!zero_node || (!IsOp(*zero_node, "Fill") && !IsOp(*zero_node, "Const"))) {
    return result;
  }

  // 3. else_value: Maximum (part of Clip)
  const NodeDef* maximum_node = FindProducer(graph, select_node.input(2));
  if (!maximum_node || !IsOp(*maximum_node, "Maximum") || maximum_node->input_size() != 2) {
    return result;
  }

  // 4. Maximum inputs: Minimum and lo
  const NodeDef* minimum_node = nullptr;
  const NodeDef* lo_node = nullptr;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* prod = FindProducer(graph, maximum_node->input(i));
    if (prod && IsOp(*prod, "Minimum")) {
      minimum_node = prod;
    } else {
      lo_node = prod;
    }
  }
  if (!minimum_node || !lo_node) return result;

  // 5. Minimum inputs: x and hi
  const NodeDef* hi_node = nullptr;
  const NodeDef* x_node = nullptr;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* prod = FindProducer(graph, minimum_node->input(i));
    if (prod && (prod->name() == isnan_node->input(0) || 
                 FusionGraphUtils::GetProducerNodeName(minimum_node->input(i)) == FusionGraphUtils::GetProducerNodeName(isnan_node->input(0)))) {
       x_node = prod;
    } else {
       hi_node = prod;
    }
  }
  if (!x_node || !hi_node) return result;

  // Success Match
  result.matched = true;
  result.matched_nodes.push_back(&select_node);
  result.matched_nodes.push_back(isnan_node);
  result.matched_nodes.push_back(zero_node);
  result.matched_nodes.push_back(maximum_node);
  result.matched_nodes.push_back(minimum_node);
  
  result.captured_nodes["x"] = x_node;
  result.captured_nodes["lo"] = lo_node;
  result.captured_nodes["hi"] = hi_node;
  result.captured_nodes["select"] = &select_node;
  
  result.captured_nodes["isnan"] = isnan_node;
  result.captured_nodes["maximum"] = maximum_node;
  result.captured_nodes["minimum"] = minimum_node;

  return result;
}

Status MusaSafeClipFusion::Apply(GraphDef* graph,
                                 const FusionMatchResult& match_result) const {
  if (!match_result.matched) {
    return errors::InvalidArgument("Invalid match result for SafeClip fusion");
  }

  const NodeDef* select_node = match_result.captured_nodes.at("select");
  const NodeDef* x_node = match_result.captured_nodes.at("x");
  const NodeDef* lo_node = match_result.captured_nodes.at("lo");
  const NodeDef* hi_node = match_result.captured_nodes.at("hi");
  
  const NodeDef* isnan_node = match_result.captured_nodes.at("isnan");
  const NodeDef* maximum_node = match_result.captured_nodes.at("maximum");
  const NodeDef* minimum_node = match_result.captured_nodes.at("minimum");

  NodeDef fused_node;
  fused_node.set_name(select_node->name());
  fused_node.set_op("MusaSafeClip");
  fused_node.set_device(select_node->device());

  // We need to find the correct input strings (including :0 etc)
  fused_node.add_input(isnan_node->input(0));
  // Find which input of maximum is minimum, and use the OTHER one as lo
  if (FusionGraphUtils::GetProducerNodeName(maximum_node->input(0)) == minimum_node->name()) {
    fused_node.add_input(maximum_node->input(1));
  } else {
    fused_node.add_input(maximum_node->input(0));
  }
  // Find which input of minimum is x, and use the OTHER one as hi
  if (FusionGraphUtils::GetProducerNodeName(minimum_node->input(0)) == x_node->name()) {
    fused_node.add_input(minimum_node->input(1));
  } else {
    fused_node.add_input(minimum_node->input(0));
  }

  if (select_node->attr().count("T")) {
    (*fused_node.mutable_attr())["T"] = select_node->attr().at("T");
  }

  // Remove nodes and add fused one
  std::set<std::string> nodes_to_remove;
  for (const auto* node : match_result.matched_nodes) {
    nodes_to_remove.insert(node->name());
  }
  
  GraphDef new_graph;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_remove.count(graph->node(i).name()) == 0) {
      *new_graph.add_node() = graph->node(i);
    }
  }
  *new_graph.add_node() = std::move(fused_node);
  graph->Swap(&new_graph);

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaSafeClipFusion);
REGISTER_FUSION_KERNEL(MusaSafeClipFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
