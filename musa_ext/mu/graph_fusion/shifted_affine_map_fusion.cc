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

#include <string>
#include <unordered_set>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"
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

// Valid op types for the variable-reading node (feeds into StridedSlice)
const std::unordered_set<std::string> kVarReadOps = {"ReadVariableOp",
                                                     "Identity", "Const"};

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool IsMaskOp(const NodeDef& node) { return kMaskOps.count(node.op()) > 0; }

bool IsVarReadOp(const NodeDef& node) {
  return kVarReadOps.count(node.op()) > 0;
}

// Find a node in the graph by its input name (handles ^ctrl and :port suffixes)
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  std::string name = FusionGraphUtils::GetProducerNodeName(input);
  if (name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, name);
}

// Check whether a StridedSlice node has a ReadVariableOp as its data source.
// StridedSlice inputs: [input, begin, end, strides]
// We verify input[0] is a ReadVariableOp (or similar variable-reading op).
bool IsStridedSliceFromVariable(const GraphDef& graph,
                                const NodeDef& slice_node) {
  if (!IsOp(slice_node, "StridedSlice")) return false;
  if (slice_node.input_size() < 1) return false;

  const NodeDef* data_src = FindProducer(graph, slice_node.input(0));
  if (!data_src) return false;

  return IsVarReadOp(*data_src);
}

// Decompose an AddV2 into (data, sliced_var) where sliced_var is a
// StridedSlice(ReadVariableOp) chain.
// Returns true if exactly one input is such a chain, and the other is the
// "data" tensor.
bool DecomposeAddV2AsDataAndSlicedVar(const GraphDef& graph,
                                      const NodeDef& add_node,
                                      const NodeDef** out_data,
                                      const NodeDef** out_sliced_var,
                                      const NodeDef** out_var_read) {
  if (add_node.input_size() < 2) return false;
  const NodeDef* in0 = FindProducer(graph, add_node.input(0));
  const NodeDef* in1 = FindProducer(graph, add_node.input(1));
  if (!in0 || !in1) return false;

  // Try both orderings (AddV2 is commutative)
  if (IsStridedSliceFromVariable(graph, *in0)) {
    *out_sliced_var = in0;
    *out_var_read = FindProducer(graph, in0->input(0));
    *out_data = in1;
    return true;
  }
  if (IsStridedSliceFromVariable(graph, *in1)) {
    *out_sliced_var = in1;
    *out_var_read = FindProducer(graph, in1->input(0));
    *out_data = in0;
    return true;
  }
  return false;
}

// Decompose a Mul node into (AddV2, mask) pair.
bool DecomposeMulAsAddMask(const GraphDef& graph, const NodeDef& mul_node,
                           const NodeDef** out_add, const NodeDef** out_mask) {
  if (mul_node.input_size() < 2) return false;
  const NodeDef* in0 = FindProducer(graph, mul_node.input(0));
  const NodeDef* in1 = FindProducer(graph, mul_node.input(1));
  if (!in0 || !in1) return false;

  // Try both orderings (Mul is commutative)
  if (IsOp(*in0, "AddV2") && IsMaskOp(*in1)) {
    *out_add = in0;
    *out_mask = in1;
    return true;
  }
  if (IsOp(*in1, "AddV2") && IsMaskOp(*in0)) {
    *out_add = in1;
    *out_mask = in0;
    return true;
  }
  return false;
}

}  // namespace

// =============================================================================
// MusaShiftedAffineMapFusion Implementation
// =============================================================================
//
// Pattern (top-down):
//   AddV2 (output)
//   ├─ Mul
//   │   ├─ AddV2 (left_add)
//   │   │   ├─ data_left
//   │   │   └─ StridedSlice ← ReadVariableOp  (sliced_var_left)
//   │   └─ Select (mask)
//   └─ AddV2 (right_add)
//       ├─ data_right
//       └─ StridedSlice ← ReadVariableOp  (sliced_var_right)
//
// Semantics:
//   output = mask * (data_left + slice(var_left))
//                  + (data_right + slice(var_right))
// =============================================================================

MusaShiftedAffineMapFusion::MusaShiftedAffineMapFusion() = default;

bool MusaShiftedAffineMapFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
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

  const NodeDef& node = graph.node(start_node_idx);

  // Entry point: output AddV2
  if (!IsOp(node, "AddV2")) {
    return FusionMatchResult{};
  }

  return MatchFromOutputAddNode(graph, start_node_idx);
}

FusionMatchResult MusaShiftedAffineMapFusion::MatchFromOutputAddNode(
    const GraphDef& graph, int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& output_add = graph.node(add_node_idx);

  VLOG(2) << "[ShiftedAffineMap::Match] ENTER, node=" << output_add.name();

  if (output_add.input_size() < 2) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: AddV2 input_size="
            << output_add.input_size() << " (need >=2)";
    return result;
  }

  const NodeDef* in0 = FindProducer(graph, output_add.input(0));
  const NodeDef* in1 = FindProducer(graph, output_add.input(1));
  if (!in0 || !in1) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: cannot resolve inputs";
    return result;
  }

  // =========================================================================
  // Identify left branch (Mul) and right branch (AddV2).
  // AddV2 is commutative, so try both orderings.
  // =========================================================================
  const NodeDef* mul_node = nullptr;
  const NodeDef* right_add_node = nullptr;

  if (IsOp(*in0, "Mul") && IsOp(*in1, "AddV2")) {
    mul_node = in0;
    right_add_node = in1;
  } else if (IsOp(*in1, "Mul") && IsOp(*in0, "AddV2")) {
    mul_node = in1;
    right_add_node = in0;
  } else {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: inputs are not (Mul, AddV2); "
            << "got (" << in0->op() << ", " << in1->op() << ")";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] Mul=" << mul_node->name()
          << ", right_add=" << right_add_node->name();

  // =========================================================================
  // Decompose Mul → (AddV2 left_add, mask)
  // =========================================================================
  const NodeDef* left_add_node = nullptr;
  const NodeDef* mask_node = nullptr;
  if (!DecomposeMulAsAddMask(graph, *mul_node, &left_add_node, &mask_node)) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: Mul is not (AddV2, mask)";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] left_add=" << left_add_node->name()
          << ", mask=" << mask_node->name();

  // =========================================================================
  // Decompose left AddV2 → (data_left, StridedSlice ← ReadVariableOp)
  // =========================================================================
  const NodeDef* data_left = nullptr;
  const NodeDef* sliced_var_left = nullptr;
  const NodeDef* var_read_left = nullptr;
  if (!DecomposeAddV2AsDataAndSlicedVar(graph, *left_add_node, &data_left,
                                        &sliced_var_left, &var_read_left)) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: left AddV2 has no "
            << "StridedSlice(ReadVariableOp) input";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] Left: data=" << data_left->name()
          << ", slice=" << sliced_var_left->name()
          << ", var=" << (var_read_left ? var_read_left->name() : "NULL");

  // =========================================================================
  // Decompose right AddV2 → (data_right, StridedSlice ← ReadVariableOp)
  // =========================================================================
  const NodeDef* data_right = nullptr;
  const NodeDef* sliced_var_right = nullptr;
  const NodeDef* var_read_right = nullptr;
  if (!DecomposeAddV2AsDataAndSlicedVar(graph, *right_add_node, &data_right,
                                        &sliced_var_right, &var_read_right)) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: right AddV2 has no "
            << "StridedSlice(ReadVariableOp) input";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] Right: data=" << data_right->name()
          << ", slice=" << sliced_var_right->name()
          << ", var=" << (var_read_right ? var_read_right->name() : "NULL");

  // =========================================================================
  // Build match result
  // =========================================================================
  result.matched = true;

  // Matched intermediate nodes (candidates for removal)
  result.matched_nodes.push_back(&output_add);
  result.matched_nodes.push_back(mul_node);
  result.matched_nodes.push_back(left_add_node);
  result.matched_nodes.push_back(right_add_node);

  // Captured nodes
  result.captured_nodes["output_add"] = &output_add;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["left_add"] = left_add_node;
  result.captured_nodes["right_add"] = right_add_node;
  result.captured_nodes["mask"] = mask_node;
  result.captured_nodes["data_left"] = data_left;
  result.captured_nodes["sliced_var_left"] = sliced_var_left;
  result.captured_nodes["var_read_left"] = var_read_left;
  result.captured_nodes["data_right"] = data_right;
  result.captured_nodes["sliced_var_right"] = sliced_var_right;
  result.captured_nodes["var_read_right"] = var_read_right;

  // Capture input edge names with :port preserved.
  // For left AddV2 → figure out which input index is which.
  for (int i = 0; i < left_add_node->input_size() && i < 2; ++i) {
    const NodeDef* producer = FindProducer(graph, left_add_node->input(i));
    if (producer == sliced_var_left)
      result.captured_attrs["sliced_var_left_input"] = left_add_node->input(i);
    else if (producer == data_left)
      result.captured_attrs["data_left_input"] = left_add_node->input(i);
  }

  // For mask from mul
  for (int i = 0; i < mul_node->input_size() && i < 2; ++i) {
    const NodeDef* producer = FindProducer(graph, mul_node->input(i));
    if (producer == mask_node)
      result.captured_attrs["mask_input"] = mul_node->input(i);
  }

  // For right AddV2
  for (int i = 0; i < right_add_node->input_size() && i < 2; ++i) {
    const NodeDef* producer = FindProducer(graph, right_add_node->input(i));
    if (producer == sliced_var_right)
      result.captured_attrs["sliced_var_right_input"] =
          right_add_node->input(i);
    else if (producer == data_right)
      result.captured_attrs["data_right_input"] = right_add_node->input(i);
  }

  VLOG(1) << "[ShiftedAffineMap::Match] SUCCESS: output_add="
          << output_add.name() << ", mul=" << mul_node->name()
          << ", left_add=" << left_add_node->name()
          << ", right_add=" << right_add_node->name()
          << ", mask=" << mask_node->name();

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
  // Extract captured information
  // -----------------------------------------------------------------------
  auto output_add_it = match_result.captured_nodes.find("output_add");
  if (output_add_it == match_result.captured_nodes.end() ||
      !output_add_it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output_add node in captured_nodes");
  }
  const NodeDef* output_add = output_add_it->second;
  const std::string output_name = output_add->name();
  const std::string output_device = output_add->device();

  // Prevent double-fusion
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaShiftedAffineMap") {
      VLOG(2) << "[ShiftedAffineMap::Apply] already fused: " << output_name;
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  // Resolve input edge names
  auto get_attr = [&](const std::string& key) -> std::string {
    auto it = match_result.captured_attrs.find(key);
    return (it != match_result.captured_attrs.end()) ? it->second : "";
  };

  std::string data_left_input = get_attr("data_left_input");
  std::string sliced_var_left_input = get_attr("sliced_var_left_input");
  std::string mask_input = get_attr("mask_input");
  std::string data_right_input = get_attr("data_right_input");
  std::string sliced_var_right_input = get_attr("sliced_var_right_input");

  if (data_left_input.empty() || sliced_var_left_input.empty() ||
      mask_input.empty() || data_right_input.empty() ||
      sliced_var_right_input.empty()) {
    VLOG(2) << "[ShiftedAffineMap::Apply] missing input edges";
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine all inputs for ShiftedAffineMap fusion");
  }

  // Determine DataType from the output AddV2 node
  DataType dtype = DT_FLOAT;
  auto dtype_it = output_add->attr().find("T");
  if (dtype_it != output_add->attr().end()) {
    dtype = dtype_it->second.type();
  }

  // -----------------------------------------------------------------------
  // Collect intermediate nodes to remove
  // -----------------------------------------------------------------------
  std::vector<std::string> nodes_to_remove;
  nodes_to_remove.push_back(output_name);  // replaced by fused node

  auto push_if_present = [&](const std::string& key) {
    auto it = match_result.captured_nodes.find(key);
    if (it != match_result.captured_nodes.end() && it->second) {
      nodes_to_remove.push_back(it->second->name());
    }
  };
  push_if_present("mul");
  push_if_present("left_add");
  push_if_present("right_add");

  // Leaf nodes kept: data_left, sliced_var_left (StridedSlice),
  // var_read_left (ReadVariableOp), mask, data_right, sliced_var_right,
  // var_read_right — they may have other consumers.

  // -----------------------------------------------------------------------
  // Remove output_add first (will be replaced by fused node)
  // -----------------------------------------------------------------------
  int output_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (output_idx >= 0) {
    FusionGraphUtils::RemoveNode(graph, output_idx);
  }

  // Remove remaining intermediate nodes only if they have no external consumers
  std::unordered_set<std::string> removable_set(nodes_to_remove.begin(),
                                                nodes_to_remove.end());
  removable_set.erase(output_name);  // already removed above
  std::vector<std::string> remaining(removable_set.begin(),
                                     removable_set.end());
  int removed = FusionGraphUtils::RemoveNodesIfUnused(graph, remaining);
  VLOG(2) << "[ShiftedAffineMap::Apply] removed " << (removed + 1)
          << " nodes (including output_add)";

  // -----------------------------------------------------------------------
  // Create fused node — reuse output_add's name
  // -----------------------------------------------------------------------
  NodeDef* fused = graph->add_node();
  fused->set_name(output_name);
  fused->set_op("MusaShiftedAffineMap");
  fused->set_device(output_device);

  // Inputs: data_left, sliced_var_left, mask, data_right, sliced_var_right
  fused->add_input(data_left_input);
  fused->add_input(sliced_var_left_input);
  fused->add_input(mask_input);
  fused->add_input(data_right_input);
  fused->add_input(sliced_var_right_input);

  // Attributes
  auto* attr = fused->mutable_attr();
  (*attr)["T"].set_type(dtype);

  VLOG(1) << "[ShiftedAffineMap::Apply] SUCCESS fused -> " << output_name
          << ", device=" << output_device;

  return Status::OK();
}

// Register fusion pattern and kernel availability
REGISTER_FUSION_PATTERN(MusaShiftedAffineMapFusion);
REGISTER_FUSION_KERNEL(MusaShiftedAffineMapFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
