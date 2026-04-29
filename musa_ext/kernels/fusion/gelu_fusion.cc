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

#include "kernels/fusion/gelu_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

constexpr float kSqrt2 = 1.41421356237f;
constexpr float kRsqrt2 = 0.70710678118f;
constexpr float kHalf = 0.5f;
constexpr float kOne = 1.0f;
constexpr float kPow3 = 3.0f;
constexpr float kApproxCoeff = 0.044715f;
constexpr float kApproxScale = 0.7978845608f;  // sqrt(2 / pi)

bool IsTruthyEnvVar(const char* env_name) {
  const char* env_val = std::getenv(env_name);
  if (env_val == nullptr) {
    return false;
  }

  const std::string value(env_val);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
         value == "YES" || value == "on" || value == "ON";
}

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
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

bool TryGetScalarFloatValue(const NodeDef& node, float* value) {
  if (!IsOp(node, "Const")) return false;

  auto it = node.attr().find("value");
  if (it == node.attr().end() || !it->second.has_tensor()) {
    return false;
  }

  const auto& tensor = it->second.tensor();
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

bool HasFloatValue(const NodeDef& node, float expected_val,
                   float tolerance = 1e-4f) {
  float actual_val = 0.0f;
  if (!TryGetScalarFloatValue(node, &actual_val)) {
    return false;
  }
  return std::abs(actual_val - expected_val) < tolerance;
}

void PushUnique(std::vector<const NodeDef*>* nodes, const NodeDef* node) {
  if (!node) return;
  auto it = std::find(nodes->begin(), nodes->end(), node);
  if (it == nodes->end()) {
    nodes->push_back(node);
  }
}

bool IsAddOp(const NodeDef& node) {
  return IsOp(node, "Add") || IsOp(node, "AddV2");
}

bool IsMulOp(const NodeDef& node) { return IsOp(node, "Mul"); }

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

bool MatchConstAndOther(const NodeDef* node, const GraphDef& graph,
                        float const_value, const NodeDef** other_input) {
  if (!node || !IsMulOp(*node) || node->input_size() != 2) {
    return false;
  }

  for (int i = 0; i < node->input_size() && i < 2; ++i) {
    const NodeDef* lhs = FindProducer(graph, node->input(i));
    const NodeDef* rhs = FindProducer(graph, node->input(1 - i));
    if (!lhs || !rhs) continue;
    if (HasFloatValue(*lhs, const_value)) {
      *other_input = rhs;
      return true;
    }
  }

  return false;
}

bool IsSqrt2Node(const NodeDef* node, const GraphDef& graph) {
  if (!node) return false;
  if (HasFloatValue(*node, kSqrt2)) {
    return true;
  }
  if (IsOp(*node, "Sqrt") && node->input_size() == 1) {
    const NodeDef* sqrt_input = FindProducer(graph, node->input(0));
    return sqrt_input && HasFloatValue(*sqrt_input, 2.0f);
  }
  return false;
}

const NodeDef* MatchScaledInput(const NodeDef* node, const GraphDef& graph) {
  if (!node) return nullptr;

  if ((IsOp(*node, "RealDiv") || IsOp(*node, "Div")) &&
      node->input_size() == 2) {
    const NodeDef* numerator = FindProducer(graph, node->input(0));
    const NodeDef* denominator = FindProducer(graph, node->input(1));
    if (numerator && denominator && !IsOp(*numerator, "Const") &&
        IsSqrt2Node(denominator, graph)) {
      return numerator;
    }
  }

  if (IsMulOp(*node) && node->input_size() == 2) {
    for (int i = 0; i < 2; ++i) {
      const NodeDef* maybe_const = FindProducer(graph, node->input(i));
      const NodeDef* maybe_input = FindProducer(graph, node->input(1 - i));
      if (!maybe_const || !maybe_input) continue;
      if (HasFloatValue(*maybe_const, kRsqrt2)) {
        return maybe_input;
      }
    }
  }

  return nullptr;
}

const NodeDef* MatchNegInput(const NodeDef* node, const GraphDef& graph) {
  if (!node || !IsOp(*node, "Neg") || node->input_size() != 1) {
    return nullptr;
  }
  return FindProducer(graph, node->input(0));
}

bool MatchExactErfFactor(const NodeDef* factor, const GraphDef& graph,
                         const NodeDef* expected_input,
                         std::vector<const NodeDef*>* matched_nodes) {
  if (!factor || !IsAddOp(*factor) || factor->input_size() != 2) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const = FindProducer(graph, factor->input(i));
    const NodeDef* maybe_erf = FindProducer(graph, factor->input(1 - i));
    if (!maybe_const || !maybe_erf) continue;

    if (!HasFloatValue(*maybe_const, kOne) || !IsOp(*maybe_erf, "Erf") ||
        maybe_erf->input_size() != 1) {
      continue;
    }

    const NodeDef* scaled_input = FindProducer(graph, maybe_erf->input(0));
    const NodeDef* original_input = MatchScaledInput(scaled_input, graph);
    if (original_input == expected_input) {
      PushUnique(matched_nodes, factor);
      PushUnique(matched_nodes, maybe_erf);
      PushUnique(matched_nodes, scaled_input);
      return true;
    }
  }

  return false;
}

bool MatchExactErfcFactor(const NodeDef* factor, const GraphDef& graph,
                          const NodeDef* expected_input,
                          std::vector<const NodeDef*>* matched_nodes) {
  if (!factor || !IsOp(*factor, "Erfc") || factor->input_size() != 1) {
    return false;
  }

  const NodeDef* scaled_neg_input = FindProducer(graph, factor->input(0));
  const NodeDef* neg_or_input = MatchScaledInput(scaled_neg_input, graph);
  const NodeDef* original_input = MatchNegInput(neg_or_input, graph);

  if (original_input == expected_input) {
    PushUnique(matched_nodes, factor);
    PushUnique(matched_nodes, scaled_neg_input);
    PushUnique(matched_nodes, neg_or_input);
    return true;
  }

  return false;
}

bool MatchPow3(const NodeDef* node, const GraphDef& graph,
               const NodeDef* expected_input,
               std::vector<const NodeDef*>* matched_nodes) {
  if (!node || !IsOp(*node, "Pow") || node->input_size() != 2) {
    return false;
  }

  const NodeDef* base = FindProducer(graph, node->input(0));
  const NodeDef* exponent = FindProducer(graph, node->input(1));
  if (base == expected_input && exponent && HasFloatValue(*exponent, kPow3)) {
    PushUnique(matched_nodes, node);
    PushUnique(matched_nodes, exponent);
    return true;
  }

  return false;
}

bool MatchApproximateFactor(const NodeDef* factor, const GraphDef& graph,
                            const NodeDef* expected_input,
                            std::vector<const NodeDef*>* matched_nodes) {
  // Keep the tanh path isolated from the exact-erf path so approximate GELU
  // can be enabled, debugged, or constrained independently later.
  if (!factor || !IsAddOp(*factor) || factor->input_size() != 2) {
    return false;
  }

  const NodeDef* tanh_node = nullptr;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const = FindProducer(graph, factor->input(i));
    const NodeDef* maybe_tanh = FindProducer(graph, factor->input(1 - i));
    if (!maybe_const || !maybe_tanh) continue;
    if (HasFloatValue(*maybe_const, kOne) && IsOp(*maybe_tanh, "Tanh")) {
      tanh_node = maybe_tanh;
      break;
    }
  }

  if (!tanh_node || tanh_node->input_size() != 1) {
    return false;
  }

  const NodeDef* tanh_scale_mul = FindProducer(graph, tanh_node->input(0));
  if (!tanh_scale_mul || !IsMulOp(*tanh_scale_mul) ||
      tanh_scale_mul->input_size() != 2) {
    return false;
  }

  const NodeDef* inner_add = nullptr;
  bool found_scale = false;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const = FindProducer(graph, tanh_scale_mul->input(i));
    const NodeDef* maybe_add =
        FindProducer(graph, tanh_scale_mul->input(1 - i));
    if (!maybe_const || !maybe_add) continue;
    if (HasFloatValue(*maybe_const, kApproxScale) && IsAddOp(*maybe_add)) {
      found_scale = true;
      inner_add = maybe_add;
      break;
    }
  }

  if (!found_scale || !inner_add || inner_add->input_size() != 2) {
    return false;
  }

  const NodeDef* cubic_mul = nullptr;
  bool found_x = false;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_input = FindProducer(graph, inner_add->input(i));
    const NodeDef* maybe_cubic_mul =
        FindProducer(graph, inner_add->input(1 - i));
    if (!maybe_input || !maybe_cubic_mul) continue;
    if (maybe_input == expected_input && IsMulOp(*maybe_cubic_mul)) {
      found_x = true;
      cubic_mul = maybe_cubic_mul;
      break;
    }
  }

  if (!found_x || !cubic_mul || cubic_mul->input_size() != 2) {
    return false;
  }

  const NodeDef* pow_node = nullptr;
  bool found_coeff = false;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* maybe_const = FindProducer(graph, cubic_mul->input(i));
    const NodeDef* maybe_pow = FindProducer(graph, cubic_mul->input(1 - i));
    if (!maybe_const || !maybe_pow) continue;
    if (HasFloatValue(*maybe_const, kApproxCoeff) &&
        MatchPow3(maybe_pow, graph, expected_input, matched_nodes)) {
      found_coeff = true;
      pow_node = maybe_pow;
      break;
    }
  }

  if (!found_coeff || !pow_node) {
    return false;
  }

  PushUnique(matched_nodes, factor);
  PushUnique(matched_nodes, tanh_node);
  PushUnique(matched_nodes, tanh_scale_mul);
  PushUnique(matched_nodes, inner_add);
  PushUnique(matched_nodes, cubic_mul);
  return true;
}

using FactorMatcher = bool (*)(const NodeDef*, const GraphDef&, const NodeDef*,
                               std::vector<const NodeDef*>*);

FusionMatchResult BuildMatchResult(const NodeDef* output_node,
                                   const NodeDef* input_node, bool approximate,
                                   const std::vector<const NodeDef*>& nodes) {
  FusionMatchResult result;
  result.matched = true;
  result.captured_nodes["output"] = output_node;
  result.captured_nodes["input"] = input_node;
  result.captured_attrs["approximate"] = approximate ? "true" : "false";
  for (const NodeDef* node : nodes) {
    PushUnique(&result.matched_nodes, node);
  }
  return result;
}

FusionMatchResult MatchFromHalfScaledInput(const GraphDef& graph,
                                           const NodeDef& final_mul,
                                           bool approximate,
                                           FactorMatcher factor_matcher) {
  if (final_mul.input_size() != 2) {
    return FusionMatchResult{};
  }

  for (int i = 0; i < final_mul.input_size() && i < 2; ++i) {
    const NodeDef* half_x_mul = FindProducer(graph, final_mul.input(i));
    const NodeDef* factor_node = FindProducer(graph, final_mul.input(1 - i));
    if (!half_x_mul || !factor_node) continue;

    const NodeDef* original_input = nullptr;
    if (!MatchConstAndOther(half_x_mul, graph, kHalf, &original_input) ||
        !original_input || IsOp(*original_input, "Const")) {
      continue;
    }

    std::vector<const NodeDef*> matched_nodes;
    PushUnique(&matched_nodes, &final_mul);
    PushUnique(&matched_nodes, half_x_mul);
    if (factor_matcher(factor_node, graph, original_input, &matched_nodes)) {
      return BuildMatchResult(&final_mul, original_input, approximate,
                              matched_nodes);
    }
  }

  return FusionMatchResult{};
}

FusionMatchResult MatchFromNestedHalfConst(const GraphDef& graph,
                                           const NodeDef& final_mul,
                                           bool approximate,
                                           FactorMatcher factor_matcher) {
  if (final_mul.input_size() != 2) {
    return FusionMatchResult{};
  }

  for (int i = 0; i < final_mul.input_size() && i < 2; ++i) {
    const NodeDef* maybe_half = FindProducer(graph, final_mul.input(i));
    const NodeDef* x_factor_mul = FindProducer(graph, final_mul.input(1 - i));
    if (!maybe_half || !x_factor_mul) continue;

    if (!HasFloatValue(*maybe_half, kHalf) || !IsMulOp(*x_factor_mul) ||
        x_factor_mul->input_size() != 2) {
      continue;
    }

    for (int j = 0; j < 2; ++j) {
      const NodeDef* original_input =
          FindProducer(graph, x_factor_mul->input(j));
      const NodeDef* factor_node =
          FindProducer(graph, x_factor_mul->input(1 - j));
      if (!original_input || !factor_node || IsOp(*original_input, "Const")) {
        continue;
      }

      std::vector<const NodeDef*> matched_nodes;
      PushUnique(&matched_nodes, &final_mul);
      PushUnique(&matched_nodes, x_factor_mul);
      PushUnique(&matched_nodes, maybe_half);
      if (factor_matcher(factor_node, graph, original_input, &matched_nodes)) {
        return BuildMatchResult(&final_mul, original_input, approximate,
                                matched_nodes);
      }
    }
  }

  return FusionMatchResult{};
}

FusionMatchResult MatchFromInputAndHalfFactor(const GraphDef& graph,
                                              const NodeDef& final_mul,
                                              bool approximate,
                                              FactorMatcher factor_matcher) {
  if (final_mul.input_size() != 2) {
    return FusionMatchResult{};
  }

  for (int i = 0; i < final_mul.input_size() && i < 2; ++i) {
    const NodeDef* original_input = FindProducer(graph, final_mul.input(i));
    const NodeDef* half_factor_mul =
        FindProducer(graph, final_mul.input(1 - i));
    if (!original_input || !half_factor_mul || IsOp(*original_input, "Const")) {
      continue;
    }

    const NodeDef* factor_node = nullptr;
    if (!MatchConstAndOther(half_factor_mul, graph, kHalf, &factor_node) ||
        !factor_node) {
      continue;
    }

    std::vector<const NodeDef*> matched_nodes;
    PushUnique(&matched_nodes, &final_mul);
    PushUnique(&matched_nodes, half_factor_mul);
    if (factor_matcher(factor_node, graph, original_input, &matched_nodes)) {
      return BuildMatchResult(&final_mul, original_input, approximate,
                              matched_nodes);
    }
  }

  return FusionMatchResult{};
}

FusionMatchResult MatchByFactor(const GraphDef& graph, int start_node_idx,
                                bool approximate,
                                FactorMatcher factor_matcher) {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& final_mul = graph.node(start_node_idx);
  if (!IsMulOp(final_mul)) {
    return FusionMatchResult{};
  }

  FusionMatchResult result =
      MatchFromHalfScaledInput(graph, final_mul, approximate, factor_matcher);
  if (result.IsValid()) {
    return result;
  }

  result =
      MatchFromNestedHalfConst(graph, final_mul, approximate, factor_matcher);
  if (result.IsValid()) {
    return result;
  }

  return MatchFromInputAndHalfFactor(graph, final_mul, approximate,
                                     factor_matcher);
}

}  // namespace

MusaGeluFusion::MusaGeluFusion() = default;

bool MusaGeluFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = !IsTruthyEnvVar("MUSA_DISABLE_GELU_FUSION");
    kernel_checked_ = true;

    if (kernel_available_) {
      VLOG(1) << "MusaGelu kernel is available";
    } else {
      VLOG(1) << "MusaGelu fusion disabled by MUSA_DISABLE_GELU_FUSION";
    }
  }
  return kernel_available_;
}

FusionMatchResult MusaGeluFusion::Match(const GraphDef& graph,
                                        int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);
  if (HasOriginalSuffix(start_node.name())) {
    return FusionMatchResult{};
  }

  // Prefer exact-erf GELU first because it is the dominant pattern in the
  // current MLP-style graphs we are targeting.
  FusionMatchResult result = MatchStandardPattern(graph, start_node_idx);
  if (result.IsValid()) {
    // VLOG(1) << "MusaGeluFusion: matched exact GELU at node "
    //         << graph.node(start_node_idx).name();
    return result;
  }

  // Keep the tanh-approximate path as an explicit fallback matcher rather
  // than interleaving it with the exact logic.
  result = MatchApproximatePattern(graph, start_node_idx);
  if (result.IsValid()) {
    // VLOG(1) << "MusaGeluFusion: matched approximate GELU at node "
    //         << graph.node(start_node_idx).name();
    return result;
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaGeluFusion::MatchStandardPattern(
    const GraphDef& graph, int start_node_idx) const {
  // Exact GELU appears either as erf(x / sqrt(2)) or as erfc(-x / sqrt(2)).
  FusionMatchResult result =
      MatchByFactor(graph, start_node_idx, false, MatchExactErfFactor);
  if (result.IsValid()) {
    return result;
  }

  return MatchByFactor(graph, start_node_idx, false, MatchExactErfcFactor);
}

FusionMatchResult MusaGeluFusion::MatchApproximatePattern(
    const GraphDef& graph, int start_node_idx) const {
  // Approximate GELU is intentionally matched in its own path.
  return MatchByFactor(graph, start_node_idx, true, MatchApproximateFactor);
}

Status MusaGeluFusion::Apply(GraphDef* graph,
                             const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid GELU match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto input_it = match_result.captured_nodes.find("input");
  if (output_it == match_result.captured_nodes.end() ||
      input_it == match_result.captured_nodes.end() || !output_it->second ||
      !input_it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in GELU pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* input_node = input_it->second;

  bool approximate = false;
  auto approx_it = match_result.captured_attrs.find("approximate");
  if (approx_it != match_result.captured_attrs.end()) {
    approximate = (approx_it->second == "true");
  }

  const std::string original_name = output_node->name();
  const std::string input_name = input_node->name();
  const std::string original_output_name = original_name + "_original";

  std::vector<std::string> removable_node_names;
  removable_node_names.reserve(match_result.matched_nodes.size());
  for (const NodeDef* matched_node : match_result.matched_nodes) {
    if (!matched_node) continue;
    if (matched_node->name() == input_name) {
      continue;
    }
    if (matched_node->name() == original_name) {
      removable_node_names.push_back(original_output_name);
    } else {
      removable_node_names.push_back(matched_node->name());
    }
  }

  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaGelu") {
      // VLOG(1) << "MusaGeluFusion: fused node already exists for "
      //         << original_name;
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
  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaGelu");
  fused_node->set_device(output_device);
  fused_node->add_input(input_name);

  auto* attr = fused_node->mutable_attr();
  if (has_output_dtype) {
    (*attr)["T"] = output_dtype;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }
  (*attr)["approximate"].set_b(approximate);

  const int removed_count = FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_node_names, {input_name, original_name});

  // VLOG(1) << "MusaGeluFusion: replaced '" << original_name
  //         << "' with MusaGelu (approximate=" << approximate
  //         << ", matched_nodes=" << match_result.matched_nodes.size()
  //         << ", removed_nodes=" << removed_count << ")";

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaGeluFusion);
REGISTER_FUSION_KERNEL(MusaGeluFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
