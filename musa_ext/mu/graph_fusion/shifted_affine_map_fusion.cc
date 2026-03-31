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

// Check whether `slice_node` is a StridedSlice whose data source is a
// ReadVariableOp (or equivalent).  StridedSlice inputs: [input, begin, end,
// strides].  We only examine input[0] (the data source).
bool IsStridedSliceFromVariable(const GraphDef& graph,
                                const NodeDef& slice_node) {
  const NodeDef* slice = ResolveIdentityLike(graph, &slice_node);
  if (!slice || !IsOp(*slice, "StridedSlice")) return false;
  if (slice->input_size() < 1) return false;
  const NodeDef* data_src = FindResolvedProducer(graph, slice->input(0));
  if (!data_src) return false;
  return IsVarReadOp(*data_src);
}

// Decompose an AddV2 into (data, sliced_var) where sliced_var is a
// StridedSlice(ReadVariableOp) chain, and data is the other input.
// Tries both orderings because AddV2 is commutative.
bool DecomposeAddV2AsDataAndSlicedVar(const GraphDef& graph,
                                      const NodeDef& add_node,
                                      const NodeDef** out_data,
                                      const NodeDef** out_sliced_var,
                                      const NodeDef** out_var_read) {
  if (add_node.input_size() < 2) return false;
  const NodeDef* in0 = FindProducer(graph, add_node.input(0));
  const NodeDef* in1 = FindProducer(graph, add_node.input(1));
  const NodeDef* in0_resolved = ResolveIdentityLike(graph, in0);
  const NodeDef* in1_resolved = ResolveIdentityLike(graph, in1);
  if (!in0 || !in1) return false;

  if (in0_resolved && IsStridedSliceFromVariable(graph, *in0_resolved)) {
    *out_sliced_var = in0;
    *out_var_read =
        FindResolvedProducer(graph, in0_resolved->input(0));
    *out_data = in1;
    return true;
  }
  if (in1_resolved && IsStridedSliceFromVariable(graph, *in1_resolved)) {
    *out_sliced_var = in1;
    *out_var_read =
        FindResolvedProducer(graph, in1_resolved->input(0));
    *out_data = in0;
    return true;
  }
  return false;
}

// Decompose a Mul node into (AddV2_left, mask).  Tries both orderings.
bool DecomposeMulAsAddMask(const GraphDef& graph, const NodeDef& mul_node,
                           const NodeDef** out_add, const NodeDef** out_mask) {
  if (mul_node.input_size() < 2) return false;
  const NodeDef* in0 = FindProducer(graph, mul_node.input(0));
  const NodeDef* in1 = FindProducer(graph, mul_node.input(1));
  const NodeDef* in0_resolved = ResolveIdentityLike(graph, in0);
  const NodeDef* in1_resolved = ResolveIdentityLike(graph, in1);
  if (!in0 || !in1) return false;

  if (in0_resolved && in1_resolved && IsOp(*in0_resolved, "AddV2") &&
      IsMaskOp(*in1_resolved)) {
    *out_add = in0;
    *out_mask = in1;
    return true;
  }
  if (in0_resolved && in1_resolved && IsOp(*in1_resolved, "AddV2") &&
      IsMaskOp(*in0_resolved)) {
    *out_add = in1;
    *out_mask = in0;
    return true;
  }
  return false;
}

}  // namespace

// =============================================================================
// MusaShiftedAffineMapFusion Implementation
//
// Pattern (top-down):
//   AddV2 (output)
//   ├─ Mul
//   │   ├─ AddV2 (left_add)
//   │   │   ├─ data_left
//   │   │   └─ StridedSlice ← ReadVariableOp   (sliced_var_left)
//   │   └─ Select (mask)
//   └─ StridedSlice ← ReadVariableOp            (sliced_var_right)
//
// Semantics:
//   output = mask * (data_left + slice(var_left)) + slice(var_right)
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

  if (output_add.input_size() < 2) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: output AddV2 has <2 inputs";
    return result;
  }

  const NodeDef* in0 = FindProducer(graph, output_add.input(0));
  const NodeDef* in1 = FindProducer(graph, output_add.input(1));
  const NodeDef* in0_resolved = ResolveIdentityLike(graph, in0);
  const NodeDef* in1_resolved = ResolveIdentityLike(graph, in1);
  if (!in0 || !in1) {
    VLOG(2)
        << "[ShiftedAffineMap::Match] FAIL: cannot resolve output_add inputs";
    return result;
  }

  // =========================================================================
  // Top AddV2: one input is Mul, the other is StridedSlice(ReadVariableOp).
  // Try both orderings.
  // =========================================================================
  const NodeDef* mul_node = nullptr;
  const NodeDef* sliced_var_right = nullptr;
  const NodeDef* var_read_right = nullptr;

  auto try_sides = [&](const NodeDef* a, const NodeDef* a_resolved,
                       const NodeDef* b, const NodeDef* b_resolved) -> bool {
    if (a_resolved && b_resolved && IsOp(*a_resolved, "Mul") &&
        IsStridedSliceFromVariable(graph, *b_resolved)) {
      mul_node = a;
      sliced_var_right = b;
      var_read_right = FindResolvedProducer(graph, b_resolved->input(0));
      return true;
    }
    return false;
  };

  if (!try_sides(in0, in0_resolved, in1, in1_resolved) &&
      !try_sides(in1, in1_resolved, in0, in0_resolved)) {
    VLOG(2) << "[ShiftedAffineMap::Match] FAIL: output AddV2 inputs are not "
            << "(Mul, StridedSlice); got ("
            << (in0_resolved ? in0_resolved->op() : "NULL") << ", "
            << (in1_resolved ? in1_resolved->op() : "NULL") << ")";
    return result;
  }

  VLOG(2) << "[ShiftedAffineMap::Match] Mul=" << mul_node->name()
          << ", sliced_var_right=" << sliced_var_right->name();

  // =========================================================================
  // Decompose Mul → (AddV2 left_add, Select mask)
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

  VLOG(2) << "[ShiftedAffineMap::Match] Left:"
          << " data=" << data_left->name()
          << ", slice=" << sliced_var_left->name()
          << ", var=" << (var_read_left ? var_read_left->name() : "NULL");

  // =========================================================================
  // Build match result
  // =========================================================================
  result.matched = true;

  // Intermediate nodes (candidates for removal)
  result.matched_nodes.push_back(&output_add);
  result.matched_nodes.push_back(mul_node);
  result.matched_nodes.push_back(left_add_node);

  // Captured nodes
  result.captured_nodes["output_add"] = &output_add;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["left_add"] = left_add_node;
  result.captured_nodes["mask"] = mask_node;
  result.captured_nodes["data_left"] = data_left;
  result.captured_nodes["sliced_var_left"] = sliced_var_left;
  result.captured_nodes["var_read_left"] = var_read_left;
  result.captured_nodes["sliced_var_right"] = sliced_var_right;
  result.captured_nodes["var_read_right"] = var_read_right;

  // Capture input edge names (with :port preserved)
  // — left_add inputs
  for (int i = 0; i < left_add_node->input_size() && i < 2; ++i) {
    const NodeDef* p = FindProducer(graph, left_add_node->input(i));
    const NodeDef* p_resolved = ResolveIdentityLike(graph, p);
    if (p == sliced_var_left)
      result.captured_attrs["sliced_var_left_input"] = left_add_node->input(i);
    else if (p == data_left)
      result.captured_attrs["data_left_input"] = left_add_node->input(i);
    else if (p_resolved && sliced_var_left &&
             p_resolved == ResolveIdentityLike(graph, sliced_var_left))
      result.captured_attrs["sliced_var_left_input"] = left_add_node->input(i);
    else if (p_resolved && data_left &&
             p_resolved == ResolveIdentityLike(graph, data_left))
      result.captured_attrs["data_left_input"] = left_add_node->input(i);
  }
  // — mask from mul
  for (int i = 0; i < mul_node->input_size() && i < 2; ++i) {
    const NodeDef* p = FindProducer(graph, mul_node->input(i));
    const NodeDef* p_resolved = ResolveIdentityLike(graph, p);
    if (p == mask_node)
      result.captured_attrs["mask_input"] = mul_node->input(i);
    else if (p_resolved && mask_node &&
             p_resolved == ResolveIdentityLike(graph, mask_node))
      result.captured_attrs["mask_input"] = mul_node->input(i);
  }
  // — sliced_var_right from output_add
  for (int i = 0; i < output_add.input_size() && i < 2; ++i) {
    const NodeDef* p = FindProducer(graph, output_add.input(i));
    const NodeDef* p_resolved = ResolveIdentityLike(graph, p);
    if (p == sliced_var_right)
      result.captured_attrs["sliced_var_right_input"] = output_add.input(i);
    else if (p_resolved && sliced_var_right &&
             p_resolved == ResolveIdentityLike(graph, sliced_var_right))
      result.captured_attrs["sliced_var_right_input"] = output_add.input(i);
  }

  VLOG(1) << "[ShiftedAffineMap::Match] SUCCESS:"
          << " output_add=" << output_add.name() << ", mul=" << mul_node->name()
          << ", left_add=" << left_add_node->name()
          << ", mask=" << mask_node->name()
          << ", sliced_var_right=" << sliced_var_right->name();

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
  std::string sliced_var_left_input = get_attr("sliced_var_left_input");
  std::string mask_input = get_attr("mask_input");
  std::string sliced_var_right_input = get_attr("sliced_var_right_input");

  if (data_left_input.empty() || sliced_var_left_input.empty() ||
      mask_input.empty() || sliced_var_right_input.empty()) {
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
  // Remove intermediate nodes (output_add, mul, left_add)
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

  // Inputs: data_left, sliced_var_left, mask, sliced_var_right  (4 inputs)
  fused->add_input(data_left_input);
  fused->add_input(sliced_var_left_input);
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
