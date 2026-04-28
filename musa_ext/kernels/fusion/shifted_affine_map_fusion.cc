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

#include "kernels/fusion/shifted_affine_map_fusion.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "kernels/fusion/fusion_pattern_manager.h"
#include "mu/optimizer/graph_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Valid op types for the mask/gate node
const std::unordered_set<std::string> kMaskOps = {"Select", "SelectV2", "Where",
                                                  "Identity"};

// Valid op types for the variable-reading node (data source of StridedSlice)
const std::unordered_set<std::string> kVarReadOps = {"ReadVariableOp",
                                                     "Identity", "Const"};

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool IsMaskOp(const NodeDef& node) { return kMaskOps.count(node.op()) > 0; }

bool IsVarReadOp(const NodeDef& node) {
  return kVarReadOps.count(node.op()) > 0;
}

// Find a producer node by input edge name (strips ^ctrl and :port suffixes)
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  std::string name = FusionGraphUtils::GetProducerNodeName(input);
  if (name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, name);
}

// Penetrate Identity nodes to find the actual op
const NodeDef* ResolveIdentityLike(const GraphDef& graph, const NodeDef* node) {
  const NodeDef* current = node;
  while (current && IsOp(*current, "Identity") && current->input_size() > 0) {
    current = FindProducer(graph, current->input(0));
  }
  return current;
}

const NodeDef* FindResolvedProducer(const GraphDef& graph,
                                    const std::string& input) {
  return ResolveIdentityLike(graph, FindProducer(graph, input));
}

}  // namespace

// =============================================================================
// MusaShiftedAffineMapFusion Implementation
//
// Pattern (top-down, post-Grappler foldable):
//   AddV2 (output)
//   ├─ Mul
//   │   ├─ const (const_left)
//   │   └─ Select (mask)
//   └─ const (const_right)
//
// Semantics:
//   output = mask * const_left + const_right
// =============================================================================

MusaShiftedAffineMapFusion::MusaShiftedAffineMapFusion() = default;

bool MusaShiftedAffineMapFusion::IsKernelAvailable() const {
  if (IsDisabledByEnv()) {
    return false;
  }
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
  // return false;
}

// 注意：这个入口函数是你上面代码遗漏的，必须保留
FusionMatchResult MusaShiftedAffineMapFusion::Match(const GraphDef& graph,
                                                    int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size())
    return FusionMatchResult{};

  const NodeDef& node = graph.node(start_node_idx);
  if (!IsOp(node, "AddV2")) return FusionMatchResult{};

  return MatchFromOutputAddNode(graph, start_node_idx);
}

FusionMatchResult MusaShiftedAffineMapFusion::MatchFromOutputAddNode(
    const GraphDef& graph, int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& output_add = graph.node(add_node_idx);

  VLOG(2) << "[ShiftedAffineMap::Match] ENTER node=" << output_add.name();

  // AddV2 必须有 2 个输入
  if (output_add.input_size() != 2) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: output AddV2 inputs != 2";
    return result;
  }

  const NodeDef* out_in0 = FindProducer(graph, output_add.input(0));
  const NodeDef* out_in1 = FindProducer(graph, output_add.input(1));
  if (!out_in0 || !out_in1) return result;

  // 内部 Helper: 校验是否为 Const
  auto is_const_from_var = [&](const NodeDef* node) -> bool {
    const NodeDef* resolved = ResolveIdentityLike(graph, node);
    return resolved && IsOp(*resolved, "Const");
  };

  // =========================================================================
  // 第 1 步: Output AddV2 -> (Mul, Const_Right)
  // =========================================================================
  const NodeDef* mul_node = nullptr;
  const NodeDef* const_right = nullptr;
  std::string const_right_input;

  if (IsOp(*out_in0, "Mul") && is_const_from_var(out_in1)) {
    mul_node = out_in0;
    const_right = FindResolvedProducer(graph, output_add.input(1));
    const_right_input = output_add.input(1);
  } else if (IsOp(*out_in1, "Mul") && is_const_from_var(out_in0)) {
    mul_node = out_in1;
    const_right = FindResolvedProducer(graph, output_add.input(0));
    const_right_input = output_add.input(0);
  } else {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL step1: output AddV2 inputs are "
               "not (Mul, Const)";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] Mul=" << mul_node->name()
          << ", const_right=" << const_right->name();

  // =========================================================================
  // 第 2 步: Mul -> (Const_Left, Select Mask)
  // =========================================================================
  if (mul_node->input_size() != 2) return result;
  const NodeDef* mul_in0 = FindProducer(graph, mul_node->input(0));
  const NodeDef* mul_in1 = FindProducer(graph, mul_node->input(1));
  if (!mul_in0 || !mul_in1) return result;

  const NodeDef* const_left = nullptr;
  const NodeDef* mask_node = nullptr;
  std::string mask_input;
  std::string const_left_input;

  if (is_const_from_var(mul_in0) && IsMaskOp(*mul_in1)) {
    const_left = FindResolvedProducer(graph, mul_node->input(0));
    const_left_input = mul_node->input(0);
    mask_node = mul_in1;
    mask_input = mul_node->input(1);
  } else if (is_const_from_var(mul_in1) && IsMaskOp(*mul_in0)) {
    const_left = FindResolvedProducer(graph, mul_node->input(1));
    const_left_input = mul_node->input(1);
    mask_node = mul_in0;
    mask_input = mul_node->input(0);
  } else {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL step2: Mul is not (Const, mask)";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] const_left=" << const_left->name()
          << ", select=" << mask_node->name();

  // =========================================================================
  // Build match result
  // =========================================================================
  result.matched = true;

  // Intermediate nodes (candidates for removal)
  result.matched_nodes.push_back(&output_add);
  result.matched_nodes.push_back(mul_node);

  // Captured nodes
  result.captured_nodes["output_add"] = &output_add;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["select"] = mask_node;
  result.captured_nodes["data_left"] = const_left;
  result.captured_nodes["sliced_var_right"] = const_right;

  // Capture input edge names directly for Apply phase
  result.captured_attrs["data_left_input"] = const_left_input;
  result.captured_attrs["mask_input"] = mask_input;
  result.captured_attrs["sliced_var_right_input"] = const_right_input;

  VLOG(1) << "[ShiftedAffineMap::Match] SUCCESS:"
          << " output_add=" << output_add.name() << ", mul=" << mul_node->name()
          << ", select=" << mask_node->name()
          << ", const_right=" << const_right->name();

  return result;
}

// =============================================================================
// Apply — replace matched sub-graph with a single MusaShiftedAffineMap node
// =============================================================================

Status MusaShiftedAffineMapFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  VLOG(2) << "[ShiftedAffineMap::Apply] ENTER";

  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid ShiftedAffineMap match result");
  }

  if (!IsKernelAvailable()) {
    VLOG(2) << "[ShiftedAffineMap::Apply] kernel not available, skipping";
    return Status::OK();
  }

  // -----------------------------------------------------------------------
  // Retrieve output node info
  // -----------------------------------------------------------------------
  auto it = match_result.captured_nodes.find("output_add");
  if (it == match_result.captured_nodes.end() || !it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output_add node in captured_nodes");
  }
  const NodeDef* output_add = it->second;
  const std::string output_name = output_add->name();
  const std::string output_device = output_add->device();

  // Prevent double-fusion
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaShiftedAffineMap") {
      VLOG(2) << "[ShiftedAffineMap::Apply] already fused: " << output_name;
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  // -----------------------------------------------------------------------
  // Resolve input edge names
  // -----------------------------------------------------------------------
  auto get_attr = [&](const std::string& key) -> std::string {
    auto a = match_result.captured_attrs.find(key);
    return (a != match_result.captured_attrs.end()) ? a->second : "";
  };

  std::string data_left_input = get_attr("data_left_input");
  std::string mask_input = get_attr("mask_input");
  std::string sliced_var_right_input = get_attr("sliced_var_right_input");

  if (data_left_input.empty() || mask_input.empty() ||
      sliced_var_right_input.empty()) {
    VLOG(2) << "[ShiftedAffineMap::Apply] FAIL: missing input edges";
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine all inputs for ShiftedAffineMap fusion");
  }

  // DataType from output AddV2
  DataType dtype = DT_FLOAT;
  {
    auto dtype_it = output_add->attr().find("T");
    if (dtype_it != output_add->attr().end()) dtype = dtype_it->second.type();
  }

  // -----------------------------------------------------------------------
  // Remove intermediate nodes (output_add, mul)
  // -----------------------------------------------------------------------
  int output_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (output_idx >= 0) FusionGraphUtils::RemoveNode(graph, output_idx);

  // Collect remaining intermediates (all except output_add, already removed)
  std::vector<std::string> remaining;
  for (const std::string& key : {"mul", "left_add"}) {
    auto nit = match_result.captured_nodes.find(key);
    if (nit != match_result.captured_nodes.end() && nit->second)
      remaining.push_back(nit->second->name());
  }
  int removed = FusionGraphUtils::RemoveNodesIfUnused(graph, remaining);
  VLOG(2) << "[ShiftedAffineMap::Apply] removed " << (removed + 1)
          << " nodes (including output_add)";

  // -----------------------------------------------------------------------
  // Create fused node — reuse output_add's name so downstream reconnects
  // -----------------------------------------------------------------------
  NodeDef* fused = graph->add_node();
  fused->set_name(output_name);
  fused->set_op("MusaShiftedAffineMap");
  fused->set_device(output_device);

  // Inputs: data_left, mask, sliced_var_right  (3 inputs)
  fused->add_input(data_left_input);
  fused->add_input(mask_input);
  fused->add_input(sliced_var_right_input);

  (*fused->mutable_attr())["T"].set_type(dtype);

  VLOG(1) << "[ShiftedAffineMap::Apply] SUCCESS -> " << output_name
          << " device=" << output_device;

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaShiftedAffineMapFusion);
REGISTER_FUSION_KERNEL(MusaShiftedAffineMapFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
