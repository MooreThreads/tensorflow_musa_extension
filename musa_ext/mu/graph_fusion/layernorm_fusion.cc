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

#include "mu/graph_fusion/layernorm_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

constexpr float kDefaultEpsilon = 1e-6f;
constexpr float kScaleOnlyUpperClip = 10000000.0f;
constexpr float kScaleOnlyLowerClip = 9.99999996e-12f;
// Reuse the existing epsilon-based LayerNorm kernel to approximate
// clip(sqrt(var), lower, upper) for the normalize-core pattern.
constexpr float kScaleOnlyEpsilon = kScaleOnlyLowerClip * kScaleOnlyLowerClip;
constexpr int64_t kReduceAxis = 1;
constexpr int64_t kKeepDimExpand = -1;
constexpr int64_t kGammaExpandDim = 0;
constexpr float kGammaBase = 1.0f;

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool IsAddOp(const NodeDef& node) {
  return IsOp(node, "Add") || IsOp(node, "AddV2");
}

bool IsDivOp(const NodeDef& node) {
  return IsOp(node, "RealDiv") || IsOp(node, "Div");
}

bool IsMulOp(const NodeDef& node) { return IsOp(node, "Mul"); }

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  const size_t colon_pos = node_name.find(':');
  if (colon_pos != std::string::npos) {
    node_name = node_name.substr(0, colon_pos);
  }

  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return &graph.node(i);
    }
  }
  return nullptr;
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

bool TryGetScalarFloatValue(const NodeDef* node, float* value) {
  if (!node || !value || !IsOp(*node, "Const")) {
    return false;
  }

  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end() || !value_it->second.has_tensor()) {
    return false;
  }

  const TensorProto& tensor = value_it->second.tensor();
  if (tensor.float_val_size() > 0) {
    *value = tensor.float_val(0);
    return true;
  }
  if (tensor.double_val_size() > 0) {
    *value = static_cast<float>(tensor.double_val(0));
    return true;
  }
  if (tensor.int_val_size() > 0) {
    *value = static_cast<float>(tensor.int_val(0));
    return true;
  }
  if (tensor.int64_val_size() > 0) {
    *value = static_cast<float>(tensor.int64_val(0));
    return true;
  }

  Tensor parsed_tensor;
  if (!parsed_tensor.FromProto(tensor) || parsed_tensor.NumElements() != 1) {
    return false;
  }

  switch (parsed_tensor.dtype()) {
    case DT_FLOAT:
      *value = parsed_tensor.flat<float>()(0);
      return true;
    case DT_DOUBLE:
      *value = static_cast<float>(parsed_tensor.flat<double>()(0));
      return true;
    case DT_INT32:
      *value = static_cast<float>(parsed_tensor.flat<int32>()(0));
      return true;
    case DT_INT64:
      *value = static_cast<float>(parsed_tensor.flat<int64>()(0));
      return true;
    default:
      return false;
  }
}

bool TryGetScalarIntValue(const NodeDef* node, int64_t* value) {
  if (!node || !value || !IsOp(*node, "Const")) {
    return false;
  }

  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end() || !value_it->second.has_tensor()) {
    return false;
  }

  const TensorProto& tensor = value_it->second.tensor();
  if (tensor.int_val_size() > 0) {
    *value = tensor.int_val(0);
    return true;
  }
  if (tensor.int64_val_size() > 0) {
    *value = tensor.int64_val(0);
    return true;
  }

  Tensor parsed_tensor;
  if (!parsed_tensor.FromProto(tensor) || parsed_tensor.NumElements() != 1) {
    return false;
  }

  switch (parsed_tensor.dtype()) {
    case DT_INT32:
      *value = parsed_tensor.flat<int32>()(0);
      return true;
    case DT_INT64:
      *value = parsed_tensor.flat<int64>()(0);
      return true;
    default:
      return false;
  }
}

bool HasFloatValue(const NodeDef* node, float expected_value,
                   float abs_tolerance, float rel_tolerance = 1e-6f) {
  float actual_value = 0.0f;
  if (!TryGetScalarFloatValue(node, &actual_value)) {
    return false;
  }

  const float tolerance =
      std::max(abs_tolerance, rel_tolerance * std::abs(expected_value));
  return std::abs(actual_value - expected_value) <= tolerance;
}

bool HasIntValue(const NodeDef* node, int64_t expected_value) {
  int64_t actual_value = 0;
  return TryGetScalarIntValue(node, &actual_value) &&
         actual_value == expected_value;
}

bool IsParameterLike(const GraphDef& graph, const NodeDef* node) {
  const NodeDef* leaf = ResolveIdentityLike(graph, node);
  if (!leaf) return false;

  return IsOp(*leaf, "Const") || IsOp(*leaf, "VariableV2") ||
         IsOp(*leaf, "VarHandleOp") || IsOp(*leaf, "ReadVariableOp");
}

void PushUnique(std::vector<const NodeDef*>* nodes, const NodeDef* node) {
  if (!node) return;
  auto it = std::find(nodes->begin(), nodes->end(), node);
  if (it == nodes->end()) {
    nodes->push_back(node);
  }
}

std::string FloatToString(float value) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<float>::max_digits10);
  oss << value;
  return oss.str();
}

bool MatchExpandDimsWithDim(const GraphDef& graph, const NodeDef* expand_node,
                            int64_t expected_dim, const NodeDef** data_input) {
  if (!expand_node || !IsOp(*expand_node, "ExpandDims") ||
      expand_node->input_size() != 2) {
    return false;
  }

  const NodeDef* dim_node = FindResolvedProducer(graph, expand_node->input(1));
  if (!HasIntValue(dim_node, expected_dim)) {
    return false;
  }

  *data_input = FindProducer(graph, expand_node->input(0));
  return *data_input != nullptr;
}

bool MatchMeanWithAxis(const GraphDef& graph, const NodeDef* mean_node,
                       int64_t expected_axis, const NodeDef** data_input) {
  if (!mean_node || !IsOp(*mean_node, "Mean") || mean_node->input_size() != 2) {
    return false;
  }

  const NodeDef* axis_node = FindResolvedProducer(graph, mean_node->input(1));
  if (!HasIntValue(axis_node, expected_axis)) {
    return false;
  }

  *data_input = FindProducer(graph, mean_node->input(0));
  return *data_input != nullptr;
}

bool MatchBinaryWithConst(const GraphDef& graph, const NodeDef* node,
                          const std::string& op_type, float expected_const,
                          float abs_tolerance, const NodeDef** other_input) {
  if (!node || !IsOp(*node, op_type) || node->input_size() != 2) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const = FindResolvedProducer(graph, node->input(i));
    const NodeDef* maybe_other = FindProducer(graph, node->input(1 - i));
    if (!maybe_const || !maybe_other) continue;

    if (HasFloatValue(maybe_const, expected_const, abs_tolerance)) {
      *other_input = maybe_other;
      return true;
    }
  }

  return false;
}

bool IsClipOp(const NodeDef& node) {
  return IsOp(node, "MusaClip") || IsOp(node, "ClipByValue") ||
         IsOp(node, "MusaClipByValue") || IsOp(node, "Clip");
}

bool MatchClipNode(const GraphDef& graph, const NodeDef* clip_node,
                   const NodeDef** unclipped_input) {
  if (!clip_node || !IsClipOp(*clip_node) || clip_node->input_size() != 3) {
    return false;
  }

  const NodeDef* lower_node = FindResolvedProducer(graph, clip_node->input(1));
  const NodeDef* upper_node = FindResolvedProducer(graph, clip_node->input(2));
  if (!HasFloatValue(lower_node, kScaleOnlyLowerClip, 1e-13f) ||
      !HasFloatValue(upper_node, kScaleOnlyUpperClip, 1e-3f)) {
    return false;
  }

  *unclipped_input = FindResolvedProducer(graph, clip_node->input(0));
  return *unclipped_input != nullptr;
}

bool MatchClippedSqrt(const GraphDef& graph, const NodeDef* clipped_node,
                      const NodeDef** sqrt_node,
                      std::vector<const NodeDef*>* matched_clip_nodes) {
  if (!clipped_node || !sqrt_node) {
    return false;
  }

  const NodeDef* matched_sqrt = nullptr;
  if (IsOp(*clipped_node, "Maximum")) {
    const NodeDef* minimum_node = nullptr;
    if (!MatchBinaryWithConst(graph, clipped_node, "Maximum",
                              kScaleOnlyLowerClip, 1e-13f, &minimum_node)) {
      return false;
    }

    if (!MatchBinaryWithConst(graph, minimum_node, "Minimum",
                              kScaleOnlyUpperClip, 1e-3f, &matched_sqrt)) {
      return false;
    }

    if (matched_clip_nodes) {
      PushUnique(matched_clip_nodes, clipped_node);
      PushUnique(matched_clip_nodes, minimum_node);
    }
  } else if (IsClipOp(*clipped_node)) {
    if (!MatchClipNode(graph, clipped_node, &matched_sqrt)) {
      return false;
    }

    if (matched_clip_nodes) {
      PushUnique(matched_clip_nodes, clipped_node);
    }
  } else {
    return false;
  }

  if (!matched_sqrt || !IsOp(*matched_sqrt, "Sqrt") ||
      matched_sqrt->input_size() != 1) {
    return false;
  }

  *sqrt_node = matched_sqrt;
  return true;
}

bool MatchCenteredSubRoot(const GraphDef& graph, const NodeDef* sub_node,
                          const NodeDef** input_node,
                          const NodeDef** mean_expand_node) {
  if (!sub_node || !IsOp(*sub_node, "Sub") || sub_node->input_size() != 2) {
    return false;
  }

  const NodeDef* lhs = FindProducer(graph, sub_node->input(0));
  const NodeDef* rhs = FindProducer(graph, sub_node->input(1));
  if (!lhs || !rhs) {
    return false;
  }

  if (!IsOp(*lhs, "ConcatV2") || !IsOp(*rhs, "ExpandDims")) {
    return false;
  }

  *input_node = lhs;
  *mean_expand_node = rhs;
  return true;
}

bool MatchCenteredSub(const GraphDef& graph, const NodeDef* sub_node,
                      const NodeDef* expected_input,
                      const NodeDef* expected_mean_expand) {
  if (!sub_node || !IsOp(*sub_node, "Sub") || sub_node->input_size() != 2) {
    return false;
  }

  const NodeDef* lhs = FindProducer(graph, sub_node->input(0));
  const NodeDef* rhs = FindProducer(graph, sub_node->input(1));
  return lhs == expected_input && rhs == expected_mean_expand;
}

bool MatchScaleGammaBranch(const GraphDef& graph, const NodeDef* gamma_add_node,
                           const NodeDef** gamma_expand_node,
                           std::string* gamma_scale_input,
                           std::string* gamma_const_input) {
  if (!gamma_add_node || !IsAddOp(*gamma_add_node) ||
      gamma_add_node->input_size() != 2) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const =
        FindResolvedProducer(graph, gamma_add_node->input(i));
    const NodeDef* maybe_expand =
        FindProducer(graph, gamma_add_node->input(1 - i));
    const NodeDef* scale_input = nullptr;
    if (!maybe_const || !maybe_expand) continue;

    if (!HasFloatValue(maybe_const, kGammaBase, 1e-6f) ||
        !MatchExpandDimsWithDim(graph, maybe_expand, kGammaExpandDim,
                                &scale_input)) {
      continue;
    }

    if (gamma_expand_node) {
      *gamma_expand_node = maybe_expand;
    }
    if (gamma_scale_input) {
      *gamma_scale_input = maybe_expand->input(0);
    }
    if (gamma_const_input) {
      *gamma_const_input = gamma_add_node->input(i);
    }
    return true;
  }

  return false;
}

}  // namespace

// =============================================================================
// MusaLayerNormFusion Implementation
// =============================================================================

MusaLayerNormFusion::MusaLayerNormFusion() = default;

bool MusaLayerNormFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaLayerNormFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);
  if (HasOriginalSuffix(start_node.name())) {
    return FusionMatchResult{};
  }

  if (IsMulOp(start_node)) {
    FusionMatchResult mul_result = MatchFromMulNode(graph, start_node_idx);
    if (mul_result.IsValid()) {
      return mul_result;
    }
  }

  if (IsAddOp(start_node)) {
    return MatchFromAddNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaLayerNormFusion::MatchFromAddNode(
    const GraphDef& graph, int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& add_node = graph.node(add_node_idx);

  if (!IsAddOp(add_node)) {
    return result;
  }

  const NodeDef* mul_node = nullptr;
  const NodeDef* beta_node = nullptr;

  for (int i = 0; i < add_node.input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, add_node.input(i));
    if (!input_node) continue;

    if (IsMulOp(*input_node)) {
      mul_node = input_node;
    } else if (IsParameterLike(graph, input_node)) {
      beta_node = input_node;
    }
  }

  if (!mul_node || !beta_node) {
    return result;
  }

  const NodeDef* div_node = nullptr;
  const NodeDef* gamma_node = nullptr;

  for (int i = 0; i < mul_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, mul_node->input(i));
    if (!input_node) continue;

    if (IsDivOp(*input_node) || IsMulOp(*input_node)) {
      div_node = input_node;
    } else if (IsParameterLike(graph, input_node)) {
      gamma_node = input_node;
    }
  }

  if (!div_node || !gamma_node) {
    return result;
  }

  const NodeDef* sub_node = nullptr;
  for (int i = 0; i < div_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, div_node->input(i));
    if (input_node && IsOp(*input_node, "Sub")) {
      sub_node = input_node;
      break;
    }
  }
  if (!sub_node) {
    return result;
  }

  const NodeDef* mean_node = nullptr;
  for (int i = 0; i < sub_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, sub_node->input(i));
    if (input_node && IsOp(*input_node, "Mean")) {
      mean_node = input_node;
      break;
    }
  }
  if (!mean_node) {
    return result;
  }

  result.matched = true;
  PushUnique(&result.matched_nodes, &add_node);
  PushUnique(&result.matched_nodes, mul_node);
  PushUnique(&result.matched_nodes, div_node);
  PushUnique(&result.matched_nodes, sub_node);
  PushUnique(&result.matched_nodes, mean_node);

  result.captured_nodes["output"] = &add_node;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["div"] = div_node;
  result.captured_nodes["sub"] = sub_node;
  result.captured_nodes["mean"] = mean_node;
  result.captured_nodes["gamma"] = gamma_node;
  result.captured_nodes["beta"] = beta_node;

  for (int i = 0; i < sub_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, sub_node->input(i));
    if (input_node && input_node != mean_node) {
      result.captured_nodes["input"] = input_node;
      result.captured_attrs["input_tensor"] = sub_node->input(i);
      break;
    }
  }

  result.captured_attrs["epsilon"] = FloatToString(kDefaultEpsilon);
  return result;
}

FusionMatchResult MusaLayerNormFusion::MatchFromMulNode(
    const GraphDef& graph, int mul_node_idx) const {
  FusionMatchResult result;
  const NodeDef& final_mul = graph.node(mul_node_idx);

  if (!IsMulOp(final_mul) || final_mul.input_size() != 2) {
    return result;
  }

  const NodeDef* realdiv_node = nullptr;
  const NodeDef* gamma_add_node = nullptr;
  const NodeDef* gamma_expand_node = nullptr;
  std::string gamma_scale_input;
  std::string gamma_const_input;

  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_div = FindProducer(graph, final_mul.input(i));
    const NodeDef* maybe_gamma = FindProducer(graph, final_mul.input(1 - i));
    if (!maybe_div || !maybe_gamma || !IsDivOp(*maybe_div)) {
      continue;
    }

    const NodeDef* matched_expand = nullptr;
    std::string matched_scale_input;
    std::string matched_const_input;
    if (!MatchScaleGammaBranch(graph, maybe_gamma, &matched_expand,
                               &matched_scale_input, &matched_const_input)) {
      continue;
    }

    realdiv_node = maybe_div;
    gamma_add_node = maybe_gamma;
    gamma_expand_node = matched_expand;
    gamma_scale_input = matched_scale_input;
    gamma_const_input = matched_const_input;
    break;
  }

  if (!realdiv_node || !gamma_add_node || !gamma_expand_node ||
      gamma_scale_input.empty() || gamma_const_input.empty()) {
    return result;
  }

  const NodeDef* sub_a = nullptr;
  const NodeDef* sqrt_node = nullptr;
  std::vector<const NodeDef*> matched_clip_nodes;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* numerator = FindProducer(graph, realdiv_node->input(i));
    const NodeDef* denominator =
        FindProducer(graph, realdiv_node->input(1 - i));
    if (!numerator || !denominator) continue;

    const NodeDef* matched_sqrt = nullptr;
    std::vector<const NodeDef*> clip_nodes;
    if (IsOp(*numerator, "Sub") &&
        MatchClippedSqrt(graph, denominator, &matched_sqrt, &clip_nodes)) {
      sub_a = numerator;
      sqrt_node = matched_sqrt;
      matched_clip_nodes = std::move(clip_nodes);
      break;
    }
  }

  if (!sub_a || !sqrt_node || matched_clip_nodes.empty()) {
    return result;
  }

  const NodeDef* concat_node = nullptr;
  const NodeDef* mean_expand_node = nullptr;
  if (!MatchCenteredSubRoot(graph, sub_a, &concat_node, &mean_expand_node)) {
    return result;
  }

  const NodeDef* mean_node = nullptr;
  if (!MatchExpandDimsWithDim(graph, mean_expand_node, kKeepDimExpand,
                              &mean_node)) {
    return result;
  }

  const NodeDef* mean_input = nullptr;
  if (!MatchMeanWithAxis(graph, mean_node, kReduceAxis, &mean_input) ||
      mean_input != concat_node || !IsOp(*concat_node, "ConcatV2")) {
    return result;
  }

  const NodeDef* var_expand_node = FindProducer(graph, sqrt_node->input(0));
  if (!var_expand_node) {
    return result;
  }

  const NodeDef* var_mean_node = nullptr;
  if (!MatchExpandDimsWithDim(graph, var_expand_node, kKeepDimExpand,
                              &var_mean_node)) {
    return result;
  }

  const NodeDef* var_mul_node = nullptr;
  if (!MatchMeanWithAxis(graph, var_mean_node, kReduceAxis, &var_mul_node) ||
      !var_mul_node || !IsMulOp(*var_mul_node) ||
      var_mul_node->input_size() != 2) {
    return result;
  }

  const NodeDef* sub_b = FindProducer(graph, var_mul_node->input(0));
  const NodeDef* sub_c = FindProducer(graph, var_mul_node->input(1));
  if (!MatchCenteredSub(graph, sub_b, concat_node, mean_expand_node) ||
      !MatchCenteredSub(graph, sub_c, concat_node, mean_expand_node)) {
    return result;
  }

  result.matched = true;
  PushUnique(&result.matched_nodes, &final_mul);
  PushUnique(&result.matched_nodes, realdiv_node);
  for (const NodeDef* clip_node : matched_clip_nodes) {
    PushUnique(&result.matched_nodes, clip_node);
  }
  PushUnique(&result.matched_nodes, sqrt_node);
  PushUnique(&result.matched_nodes, var_expand_node);
  PushUnique(&result.matched_nodes, var_mean_node);
  PushUnique(&result.matched_nodes, var_mul_node);
  PushUnique(&result.matched_nodes, sub_a);
  PushUnique(&result.matched_nodes, sub_b);
  PushUnique(&result.matched_nodes, sub_c);
  PushUnique(&result.matched_nodes, mean_expand_node);
  PushUnique(&result.matched_nodes, mean_node);
  PushUnique(&result.matched_nodes, gamma_add_node);
  PushUnique(&result.matched_nodes, gamma_expand_node);

  result.captured_nodes["output"] = &final_mul;
  result.captured_nodes["input"] = concat_node;
  result.captured_nodes["div"] = realdiv_node;
  result.captured_nodes["sub"] = sub_a;
  result.captured_nodes["mean"] = mean_node;

  result.captured_attrs["input_tensor"] = sub_a->input(0);
  result.captured_attrs["gamma_scale_input"] = gamma_scale_input;
  result.captured_attrs["gamma_const_input"] = gamma_const_input;
  result.captured_attrs["epsilon"] = FloatToString(kScaleOnlyEpsilon);

  return result;
}

Status MusaLayerNormFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid LayerNorm match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto input_it = match_result.captured_nodes.find("input");
  if (output_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output node in LayerNorm pattern");
  }

  const NodeDef* output_node = output_it->second;
  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaLayerNorm") {
      VLOG(1) << "MusaLayerNorm: Output node " << original_name
              << " is already fused, skipping";
      return Status::OK();
    }
  }

  int output_node_idx = -1;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == original_name) {
      output_node_idx = i;
      break;
    }
  }
  if (output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find output node in graph: " + original_name);
  }

  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_output_node->device();
  AttrValue output_dtype;
  const auto dtype_it = original_output_node->attr().find("T");
  const bool has_output_dtype = dtype_it != original_output_node->attr().end();
  if (has_output_dtype) {
    output_dtype = dtype_it->second;
  }

  float epsilon = kDefaultEpsilon;
  auto epsilon_it = match_result.captured_attrs.find("epsilon");
  if (epsilon_it != match_result.captured_attrs.end()) {
    epsilon = std::stof(epsilon_it->second);
  }

  std::string fused_input_name;
  auto input_name_it = match_result.captured_attrs.find("input_tensor");
  if (input_name_it != match_result.captured_attrs.end()) {
    fused_input_name = input_name_it->second;
  } else if (input_it != match_result.captured_nodes.end() &&
             input_it->second) {
    fused_input_name = input_it->second->name();
  } else {
    auto mean_it = match_result.captured_nodes.find("mean");
    if (mean_it != match_result.captured_nodes.end() && mean_it->second &&
        mean_it->second->input_size() > 0) {
      fused_input_name = mean_it->second->input(0);
    } else {
      return Status(error::INVALID_ARGUMENT,
                    "Cannot determine LayerNorm input");
    }
  }

  const NodeDef* gamma_node = nullptr;
  const NodeDef* beta_node = nullptr;

  auto gamma_it = match_result.captured_nodes.find("gamma");
  if (gamma_it != match_result.captured_nodes.end()) {
    gamma_node = gamma_it->second;
  }
  auto beta_it = match_result.captured_nodes.find("beta");
  if (beta_it != match_result.captured_nodes.end()) {
    beta_node = beta_it->second;
  }

  std::string gamma_name;
  auto gamma_scale_it = match_result.captured_attrs.find("gamma_scale_input");
  auto gamma_const_it = match_result.captured_attrs.find("gamma_const_input");
  if (gamma_scale_it != match_result.captured_attrs.end() &&
      gamma_const_it != match_result.captured_attrs.end()) {
    gamma_name = original_name + "/fused_gamma";
    if (FusionGraphUtils::FindNodeIndex(*graph, gamma_name) < 0) {
      NodeDef* gamma_add_node = graph->add_node();
      gamma_add_node->set_name(gamma_name);
      gamma_add_node->set_op("AddV2");
      gamma_add_node->set_device(output_device);
      gamma_add_node->add_input(gamma_const_it->second);
      gamma_add_node->add_input(gamma_scale_it->second);

      auto* gamma_attr = gamma_add_node->mutable_attr();
      if (has_output_dtype) {
        (*gamma_attr)["T"] = output_dtype;
      } else {
        (*gamma_attr)["T"].set_type(DT_FLOAT);
      }
    }
  } else if (gamma_node) {
    gamma_name = gamma_node->name();
  } else {
    return Status(error::INVALID_ARGUMENT,
                  "Missing gamma input in LayerNorm fusion");
  }

  std::string beta_name;
  if (beta_node) {
    beta_name = beta_node->name();
  } else {
    beta_name = original_name + "/fused_beta";
    if (FusionGraphUtils::FindNodeIndex(*graph, beta_name) < 0) {
      NodeDef* beta_zero_node = graph->add_node();
      beta_zero_node->set_name(beta_name);
      beta_zero_node->set_op("ZerosLike");
      beta_zero_node->set_device(output_device);
      beta_zero_node->add_input(gamma_name);

      auto* beta_attr = beta_zero_node->mutable_attr();
      if (has_output_dtype) {
        (*beta_attr)["T"] = output_dtype;
      } else {
        (*beta_attr)["T"].set_type(DT_FLOAT);
      }
    }
  }

  std::vector<std::string> removable_node_names;
  removable_node_names.reserve(match_result.matched_nodes.size());
  const std::string input_name =
      (input_it != match_result.captured_nodes.end() && input_it->second)
          ? input_it->second->name()
          : "";
  for (const NodeDef* matched_node : match_result.matched_nodes) {
    if (!matched_node) continue;
    if (!input_name.empty() && matched_node->name() == input_name) {
      continue;
    }
    if (matched_node->name() == original_name) {
      removable_node_names.push_back(original_output_name);
    } else {
      removable_node_names.push_back(matched_node->name());
    }
  }

  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaLayerNorm");
  fused_node->set_device(output_device);
  fused_node->add_input(fused_input_name);
  fused_node->add_input(gamma_name);
  fused_node->add_input(beta_name);

  auto* attr = fused_node->mutable_attr();
  if (has_output_dtype) {
    (*attr)["T"] = output_dtype;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }
  (*attr)["epsilon"].set_f(epsilon);

  std::unordered_set<std::string> protected_node_names = {original_name};
  auto protect_input = [&protected_node_names](const std::string& input_name) {
    const std::string producer =
        FusionGraphUtils::GetProducerNodeName(input_name);
    if (!producer.empty()) {
      protected_node_names.insert(producer);
    }
  };
  protect_input(fused_input_name);
  protect_input(gamma_name);
  protect_input(beta_name);

  const int removed_count = FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_node_names, protected_node_names);

  VLOG(1) << "MusaLayerNorm: Replaced '" << original_name
          << "' with MusaLayerNorm (removed_nodes=" << removed_count << ")";

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaLayerNormFusion);
REGISTER_FUSION_KERNEL(MusaLayerNormFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
