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

#include "mu/graph_fusion/shifted_affine_map_fusion.h"

#include <fstream>
#include <initializer_list>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

constexpr char kShiftedAffineMapFusionLogPath[] =
    "/tmp/musa_shifted_affine_map_fusion.log";

struct ConstInputMatch {
  bool matched = false;
  const NodeDef* passthrough_node = nullptr;
  const NodeDef* const_node = nullptr;
  std::string input_to_use;
};

void AppendFusionLog(const std::string& message) {
  std::ofstream log(kShiftedAffineMapFusionLogPath, std::ios::app);
  if (log.is_open() == false) {
    return;
  }
  log << message << '\n';
}

void LogFoundNode(const char* stage, const NodeDef* node) {
  if (node == nullptr) {
    return;
  }
  AppendFusionLog(std::string(stage) + ": " + node->name() + "(" +
                  node->op() + ")");
}

bool IsAnyOp(const NodeDef& node,
             std::initializer_list<const char*> op_types) {
  for (const char* op_type : op_types) {
    if (node.op() == op_type) {
      return true;
    }
  }
  return false;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string node_name = FusionGraphUtils::GetProducerNodeName(input);
  if (node_name.empty()) {
    return nullptr;
  }
  return FusionGraphUtils::GetNodeByName(graph, node_name);
}

std::string GetCapturedAttr(const FusionMatchResult& match_result,
                            const std::string& key) {
  auto it = match_result.captured_attrs.find(key);
  if (it == match_result.captured_attrs.end()) {
    return "";
  }
  return it->second;
}

void AddUniqueMatchedNode(FusionMatchResult* result, const NodeDef* node) {
  if (node == nullptr) {
    return;
  }
  for (const NodeDef* existing : result->matched_nodes) {
    if (existing == node) {
      return;
    }
  }
  result->matched_nodes.push_back(node);
}

ConstInputMatch MatchConstInput(const GraphDef& graph,
                                const std::string& input,
                                const char* stage) {
  ConstInputMatch match;
  match.input_to_use = input;

  const NodeDef* producer = FindProducer(graph, input);
  if (producer == nullptr) {
    return match;
  }

  if (IsAnyOp(*producer, {"Identity", "Snapshot"})) {
    match.passthrough_node = producer;
    if (producer->input_size() == 1) {
      match.input_to_use = producer->input(0);
      producer = FindProducer(graph, producer->input(0));
    } else {
      return ConstInputMatch{};
    }
    if (producer == nullptr) {
      return ConstInputMatch{};
    }
  }

  if (producer->op() == "Const") {
    match.matched = true;
    match.const_node = producer;
    LogFoundNode(stage, producer);
    if (match.passthrough_node != nullptr) {
      LogFoundNode("matched const passthrough", match.passthrough_node);
    }
    return match;
  }

  return ConstInputMatch{};
}

FusionMatchResult TryMatchShiftedAffineMapFromAdd(
    const GraphDef& graph, const NodeDef& output_add,
    const NodeDef& select_node, int select_input_index) {
  FusionMatchResult result;

  if (IsAnyOp(output_add, {"Add", "AddV2"}) == false ||
      output_add.input_size() != 2) {
    return result;
  }
  LogFoundNode("matched output add", &output_add);

  for (int mul_input_index = 0; mul_input_index < 2; ++mul_input_index) {
    const int const2_input_index = 1 - mul_input_index;
    const NodeDef* mul_node =
        FindProducer(graph, output_add.input(mul_input_index));
    if (mul_node == nullptr || mul_node->op() != "Mul" ||
        mul_node->input_size() != 2) {
      continue;
    }
    LogFoundNode("matched mul", mul_node);

    ConstInputMatch const2_match =
        MatchConstInput(graph, output_add.input(const2_input_index),
                        "matched const2");
    if (const2_match.matched == false) {
      continue;
    }

    for (int const1_input_index = 0; const1_input_index < 2;
         ++const1_input_index) {
      const int x_input_index = 1 - const1_input_index;
      ConstInputMatch const1_match =
          MatchConstInput(graph, mul_node->input(const1_input_index),
                          "matched const1");
      if (const1_match.matched == false) {
        continue;
      }

      const std::string x_input = mul_node->input(x_input_index);
      if (x_input.empty()) {
        continue;
      }

      const NodeDef* x_producer = FindProducer(graph, x_input);
      if (x_producer != nullptr) {
        LogFoundNode("matched select input producer", x_producer);
      } else {
        AppendFusionLog("matched select input tensor: " + x_input);
      }

      result.matched = true;
      result.captured_nodes["select"] = &select_node;
      result.captured_nodes["output_add"] = &output_add;
      result.captured_nodes["mul"] = mul_node;
      result.captured_nodes["const1"] = const1_match.const_node;
      result.captured_nodes["const2"] = const2_match.const_node;

      result.captured_attrs["select_input_index"] =
          std::to_string(select_input_index);
      result.captured_attrs["x_input"] = x_input;
      result.captured_attrs["const1_input"] = const1_match.input_to_use;
      result.captured_attrs["const2_input"] = const2_match.input_to_use;

      AddUniqueMatchedNode(&result, &output_add);
      AddUniqueMatchedNode(&result, mul_node);
      AddUniqueMatchedNode(&result, const1_match.passthrough_node);
      AddUniqueMatchedNode(&result, const2_match.passthrough_node);

      AppendFusionLog(
          "FULL MATCH: select=" + select_node.name() +
          ", branch_input=" + std::to_string(select_input_index) +
          ", add=" + output_add.name() + ", mul=" + mul_node->name() +
          ", const1=" + const1_match.const_node->name() +
          ", const2=" + const2_match.const_node->name());
      return result;
    }
  }

  return FusionMatchResult{};
}

}  // namespace

MusaShiftedAffineMapFusion::MusaShiftedAffineMapFusion() = default;

bool MusaShiftedAffineMapFusion::IsKernelAvailable() const {
  if (kernel_checked_ == false) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaShiftedAffineMapFusion::Match(const GraphDef& graph,
                                                    int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);
  if (IsAnyOp(start_node, {"Select", "SelectV2"}) == false) {
    return FusionMatchResult{};
  }

  AppendFusionLog("ENTER MATCH from select: " + start_node.name());
  LogFoundNode("matched start select", &start_node);
  return MatchFromSelectNode(graph, start_node_idx);
}

FusionMatchResult MusaShiftedAffineMapFusion::MatchFromSelectNode(
    const GraphDef& graph, int select_node_idx) const {
  FusionMatchResult result;
  const NodeDef& select_node = graph.node(select_node_idx);

  if (IsAnyOp(select_node, {"Select", "SelectV2"}) == false ||
      select_node.input_size() != 3) {
    return result;
  }

  for (int input_index = 1; input_index <= 2; ++input_index) {
    const NodeDef* output_add =
        FindProducer(graph, select_node.input(input_index));
    if (output_add == nullptr) {
      continue;
    }

    result = TryMatchShiftedAffineMapFromAdd(graph, *output_add, select_node,
                                             input_index);
    if (result.matched) {
      return result;
    }
  }

  return FusionMatchResult{};
}

Status MusaShiftedAffineMapFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (match_result.IsValid() == false) {
    AppendFusionLog("APPLY FAIL: invalid match result");
    return Status(error::INVALID_ARGUMENT,
                  "Invalid ShiftedAffineMap match result");
  }

  if (IsKernelAvailable() == false) {
    AppendFusionLog("APPLY SKIP: kernel unavailable");
    return Status::OK();
  }

  auto select_it = match_result.captured_nodes.find("select");
  auto output_add_it = match_result.captured_nodes.find("output_add");
  if (select_it == match_result.captured_nodes.end() ||
      output_add_it == match_result.captured_nodes.end()) {
    AppendFusionLog("APPLY FAIL: missing select or output add node");
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured nodes for ShiftedAffineMap fusion");
  }
  if (select_it->second == nullptr || output_add_it->second == nullptr) {
    AppendFusionLog("APPLY FAIL: null select or output add node");
    return Status(error::INVALID_ARGUMENT,
                  "Null captured nodes for ShiftedAffineMap fusion");
  }

  const std::string select_name = select_it->second->name();
  const std::string output_add_name = output_add_it->second->name();
  const std::string fused_node_name =
      output_add_name + "_musa_shifted_affine_map";

  const std::string x_input = GetCapturedAttr(match_result, "x_input");
  const std::string const1_input =
      GetCapturedAttr(match_result, "const1_input");
  const std::string const2_input =
      GetCapturedAttr(match_result, "const2_input");
  const std::string select_input_index_str =
      GetCapturedAttr(match_result, "select_input_index");
  if (x_input.empty() || const1_input.empty() || const2_input.empty() ||
      select_input_index_str.empty()) {
    AppendFusionLog("APPLY FAIL: missing fused inputs or branch index");
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured attributes for ShiftedAffineMap fusion");
  }

  const int select_node_idx =
      FusionGraphUtils::FindNodeIndex(*graph, select_name);
  if (select_node_idx < 0) {
    AppendFusionLog("APPLY FAIL: select node not found in mutable graph");
    return Status(error::NOT_FOUND, "Select node not found");
  }

  int select_input_index = 0;
  try {
    select_input_index = std::stoi(select_input_index_str);
  } catch (...) {
    AppendFusionLog("APPLY FAIL: invalid select input index");
    return Status(error::INVALID_ARGUMENT, "Invalid select input index");
  }

  if (select_input_index < 1 || select_input_index > 2) {
    AppendFusionLog("APPLY FAIL: select input index out of range");
    return Status(error::INVALID_ARGUMENT,
                  "Select input index out of range");
  }

  for (const NodeDef& node : graph->node()) {
    if (node.name() == fused_node_name &&
        node.op() == "MusaShiftedAffineMap") {
      AppendFusionLog("APPLY SKIP: fused node already exists " +
                      fused_node_name);
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  DataType dtype = DT_FLOAT;
  auto dtype_it = output_add_it->second->attr().find("T");
  if (dtype_it != output_add_it->second->attr().end()) {
    dtype = dtype_it->second.type();
  }

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaShiftedAffineMap");
  fused_node->set_device(output_add_it->second->device());
  fused_node->add_input(x_input);
  fused_node->add_input(const1_input);
  fused_node->add_input(const2_input);
  (*fused_node->mutable_attr())["T"].set_type(dtype);

  graph->mutable_node(select_node_idx)->set_input(select_input_index,
                                                  fused_node_name);
  AppendFusionLog("APPLY REWIRE: select=" + select_name +
                  ", branch_input=" + std::to_string(select_input_index) +
                  ", new_node=" + fused_node_name);

  std::vector<std::string> cleanup_nodes;
  std::unordered_set<std::string> seen_names;
  for (const NodeDef* node : match_result.matched_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (seen_names.insert(node->name()).second) {
      cleanup_nodes.push_back(node->name());
    }
  }

  const int removed_count =
      FusionGraphUtils::RemoveNodesIfUnused(graph, cleanup_nodes);

  AppendFusionLog("APPLY SUCCESS: created " + fused_node_name +
                  ", removed_nodes=" + std::to_string(removed_count));
  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaShiftedAffineMapFusion);
REGISTER_FUSION_KERNEL(MusaShiftedAffineMapFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
