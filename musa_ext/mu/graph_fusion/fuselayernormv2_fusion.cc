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

#include "mu/graph_fusion/fuselayernormv2_fusion.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

constexpr float kDefaultEpsilon = 0.001f;

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
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

bool HasStringAttr(const NodeDef& node, const std::string& attr_name,
                   const std::string& expected_value) {
  const auto it = node.attr().find(attr_name);
  return it != node.attr().end() && it->second.s() == expected_value;
}

bool HasBoolAttr(const NodeDef& node, const std::string& attr_name,
                 bool expected_value) {
  const auto it = node.attr().find(attr_name);
  return it != node.attr().end() && it->second.b() == expected_value;
}

bool GetFloatAttr(const NodeDef& node, const std::string& attr_name,
                  float* value) {
  if (!value) return false;
  const auto it = node.attr().find(attr_name);
  if (it == node.attr().end()) return false;
  *value = it->second.f();
  return true;
}

bool IsParameterLike(const GraphDef& graph, const NodeDef* node) {
  const NodeDef* leaf = ResolveIdentityLike(graph, node);
  if (!leaf) return false;

  return IsOp(*leaf, "Const") || IsOp(*leaf, "VariableV2") ||
         IsOp(*leaf, "VarHandleOp") || IsOp(*leaf, "ReadVariableOp") ||
         IsOp(*leaf, "Placeholder");
}

std::string GetResolvedProducerName(const GraphDef& graph,
                                    const std::string& input) {
  const NodeDef* producer = FindResolvedProducer(graph, input);
  if (producer) {
    return producer->name();
  }
  return FusionGraphUtils::GetProducerNodeName(input);
}

const NodeDef* UnpackSingleElementPack(const GraphDef& graph,
                                       const NodeDef* node) {
  const NodeDef* resolved = ResolveIdentityLike(graph, node);
  if (!resolved) {
    return nullptr;
  }
  if (!IsOp(*resolved, "Pack") || resolved->input_size() != 1) {
    return resolved;
  }
  return FindResolvedProducer(graph, resolved->input(0));
}

const NodeDef* UnwrapExpandDimsAxisZero(const GraphDef& graph,
                                        const NodeDef* node) {
  const NodeDef* resolved = ResolveIdentityLike(graph, node);
  if (!resolved || !IsOp(*resolved, "ExpandDims") ||
      resolved->input_size() != 2) {
    return resolved;
  }

  const NodeDef* axis_node = FindResolvedProducer(graph, resolved->input(1));
  if (!HasIntValue(axis_node, 0)) {
    return resolved;
  }

  return FindResolvedProducer(graph, resolved->input(0));
}

bool AreEquivalentDimsNode(const GraphDef& graph, const NodeDef* a,
                           const NodeDef* b) {
  if (!a || !b) {
    return false;
  }
  if (a == b) {
    return true;
  }

  const NodeDef* resolved_a = ResolveIdentityLike(graph, a);
  const NodeDef* resolved_b = ResolveIdentityLike(graph, b);
  if (!resolved_a || !resolved_b) {
    return false;
  }
  if (resolved_a == resolved_b) {
    return true;
  }

  const NodeDef* unpacked_a = UnpackSingleElementPack(graph, resolved_a);
  const NodeDef* unpacked_b = UnpackSingleElementPack(graph, resolved_b);
  if (unpacked_a && unpacked_b && unpacked_a == unpacked_b) {
    return true;
  }

  const NodeDef* normalized_a =
      UnwrapExpandDimsAxisZero(graph, unpacked_a ? unpacked_a : resolved_a);
  const NodeDef* normalized_b =
      UnwrapExpandDimsAxisZero(graph, unpacked_b ? unpacked_b : resolved_b);
  if (normalized_a && normalized_b && normalized_a == normalized_b) {
    return true;
  }

  if (!IsOp(*resolved_a, "Pack") || !IsOp(*resolved_b, "Pack") ||
      resolved_a->input_size() != resolved_b->input_size()) {
    return false;
  }

  for (int i = 0; i < resolved_a->input_size(); ++i) {
    if (GetResolvedProducerName(graph, resolved_a->input(i)) !=
        GetResolvedProducerName(graph, resolved_b->input(i))) {
      return false;
    }
  }
  return true;
}

bool MatchReshapeInputShape(const GraphDef& graph, const NodeDef* reshape_dims,
                            const NodeDef* expected_batch_dims) {
  const NodeDef* resolved_shape = ResolveIdentityLike(graph, reshape_dims);
  if (!resolved_shape || !IsOp(*resolved_shape, "Pack") ||
      resolved_shape->input_size() != 4) {
    return false;
  }

  const NodeDef* first_dim = FindResolvedProducer(graph, resolved_shape->input(0));
  const NodeDef* second_dim =
      FindResolvedProducer(graph, resolved_shape->input(1));
  const NodeDef* last_dim = FindResolvedProducer(graph, resolved_shape->input(3));
  if (!HasIntValue(first_dim, 1) || !HasIntValue(last_dim, 1) || !second_dim) {
    return false;
  }

  return AreEquivalentDimsNode(graph, second_dim, expected_batch_dims);
}

bool MatchFillWithValue(const GraphDef& graph, const NodeDef* fill_node,
                        float expected_value, const NodeDef** dims_node) {
  if (!fill_node || !IsOp(*fill_node, "Fill") || fill_node->input_size() != 2) {
    return false;
  }

  const NodeDef* value_node = FindResolvedProducer(graph, fill_node->input(1));
  if (!HasFloatValue(value_node, expected_value, 1e-6f)) {
    return false;
  }

  if (dims_node) {
    *dims_node = FindResolvedProducer(graph, fill_node->input(0));
  }
  return true;
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

}  // namespace

MusaFuseLayerNormV2Fusion::MusaFuseLayerNormV2Fusion() = default;

bool MusaFuseLayerNormV2Fusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaFuseLayerNormV2Fusion::Match(const GraphDef& graph,
                                                   int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);
  if (HasOriginalSuffix(start_node.name()) || !IsAddOp(start_node)) {
    return FusionMatchResult{};
  }

  return MatchFromAddNode(graph, start_node_idx);
}

FusionMatchResult MusaFuseLayerNormV2Fusion::MatchFromAddNode(
    const GraphDef& graph, int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& add_node = graph.node(add_node_idx);
  if (!IsAddOp(add_node) || add_node.input_size() != 2) {
    return result;
  }

  const NodeDef* mul_node = nullptr;
  std::string beta_tensor;

  for (int i = 0; i < 2; ++i) {
    const NodeDef* input_node = FindResolvedProducer(graph, add_node.input(i));
    if (!input_node) continue;

    if (IsMulOp(*input_node)) {
      mul_node = input_node;
    } else if (IsParameterLike(graph, input_node)) {
      beta_tensor = add_node.input(i);
    }
  }

  if (!mul_node || beta_tensor.empty() || mul_node->input_size() != 2) {
    return result;
  }

  const NodeDef* reshape_out_node = nullptr;
  std::string gamma_tensor;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* input_node = FindResolvedProducer(graph, mul_node->input(i));
    if (!input_node) continue;

    if (IsOp(*input_node, "Reshape")) {
      reshape_out_node = input_node;
    } else if (IsParameterLike(graph, input_node)) {
      gamma_tensor = mul_node->input(i);
    }
  }

  if (!reshape_out_node || gamma_tensor.empty() ||
      reshape_out_node->input_size() != 2) {
    return result;
  }

  const NodeDef* bn_node =
      FindResolvedProducer(graph, reshape_out_node->input(0));
  if (!bn_node || !IsOp(*bn_node, "FusedBatchNormV3") ||
      bn_node->input_size() < 3 ||
      !HasStringAttr(*bn_node, "data_format", "NCHW") ||
      !HasBoolAttr(*bn_node, "is_training", true)) {
    return result;
  }

  const NodeDef* shape_node =
      FindResolvedProducer(graph, reshape_out_node->input(1));
  if (!shape_node || !IsOp(*shape_node, "Shape") || shape_node->input_size() != 1) {
    return result;
  }

  const NodeDef* reshape_in_node = FindResolvedProducer(graph, bn_node->input(0));
  if (!reshape_in_node || !IsOp(*reshape_in_node, "Reshape") ||
      reshape_in_node->input_size() != 2) {
    return result;
  }

  const NodeDef* input_node =
      FindResolvedProducer(graph, reshape_in_node->input(0));
  const NodeDef* shape_input_node =
      FindResolvedProducer(graph, shape_node->input(0));
  if (!input_node || !shape_input_node || input_node != shape_input_node) {
    return result;
  }

  const NodeDef* scale_fill = FindResolvedProducer(graph, bn_node->input(1));
  const NodeDef* offset_fill = FindResolvedProducer(graph, bn_node->input(2));
  const NodeDef* scale_dims = nullptr;
  const NodeDef* offset_dims = nullptr;
  if (!MatchFillWithValue(graph, scale_fill, 1.0f, &scale_dims) ||
      !MatchFillWithValue(graph, offset_fill, 0.0f, &offset_dims) ||
      !scale_dims || !offset_dims ||
      !AreEquivalentDimsNode(graph, scale_dims, offset_dims)) {
    return result;
  }

  const NodeDef* reshape_dims =
      FindResolvedProducer(graph, reshape_in_node->input(1));
  if (!reshape_dims ||
      !MatchReshapeInputShape(graph, reshape_dims, scale_dims)) {
    return result;
  }

  float epsilon = kDefaultEpsilon;
  GetFloatAttr(*bn_node, "epsilon", &epsilon);

  result.matched = true;
  PushUnique(&result.matched_nodes, &add_node);
  PushUnique(&result.matched_nodes, mul_node);
  PushUnique(&result.matched_nodes, reshape_out_node);
  PushUnique(&result.matched_nodes, bn_node);
  PushUnique(&result.matched_nodes, reshape_in_node);
  PushUnique(&result.matched_nodes, shape_node);
  PushUnique(&result.matched_nodes, scale_fill);
  PushUnique(&result.matched_nodes, offset_fill);
  PushUnique(&result.matched_nodes, scale_dims);
  PushUnique(&result.matched_nodes, offset_dims);
  PushUnique(&result.matched_nodes, reshape_dims);

  result.captured_nodes["output"] = &add_node;
  result.captured_nodes["input"] = input_node;
  result.captured_nodes["gamma"] = FindResolvedProducer(graph, gamma_tensor);
  result.captured_nodes["beta"] = FindResolvedProducer(graph, beta_tensor);

  result.captured_attrs["input_tensor"] = reshape_in_node->input(0);
  result.captured_attrs["gamma_tensor"] = gamma_tensor;
  result.captured_attrs["beta_tensor"] = beta_tensor;
  result.captured_attrs["epsilon"] = FloatToString(epsilon);

  LOG(INFO) << "[FuseLayerNormV2][Match] matched subgraph at output="
            << add_node.name() << ", bn=" << bn_node->name()
            << ", reshape_in=" << reshape_in_node->name()
            << ", reshape_out=" << reshape_out_node->name();

  return result;
}

Status MusaFuseLayerNormV2Fusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid FuseLayerNormV2 match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  if (output_it == match_result.captured_nodes.end() || !output_it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output node in FuseLayerNormV2 pattern");
  }

  auto input_tensor_it = match_result.captured_attrs.find("input_tensor");
  auto gamma_tensor_it = match_result.captured_attrs.find("gamma_tensor");
  auto beta_tensor_it = match_result.captured_attrs.find("beta_tensor");
  if (input_tensor_it == match_result.captured_attrs.end() ||
      gamma_tensor_it == match_result.captured_attrs.end() ||
      beta_tensor_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing tensors in FuseLayerNormV2 pattern");
  }

  const std::string input_tensor = input_tensor_it->second;
  const std::string gamma_tensor = gamma_tensor_it->second;
  const std::string beta_tensor = beta_tensor_it->second;

  const NodeDef* output_node = output_it->second;
  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  LOG(INFO) << "[FuseLayerNormV2][Apply] start apply for output="
            << original_name << ", input=" << input_tensor
            << ", gamma=" << gamma_tensor << ", beta=" << beta_tensor;

  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaLayerNorm") {
      VLOG(1) << "FuseLayerNormV2: Output node " << original_name
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

  std::vector<std::string> removable_node_names;
  removable_node_names.reserve(match_result.matched_nodes.size());
  for (const NodeDef* matched_node : match_result.matched_nodes) {
    if (!matched_node) continue;
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
  fused_node->add_input(input_tensor);
  fused_node->add_input(gamma_tensor);
  fused_node->add_input(beta_tensor);

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
  protect_input(input_tensor);
  protect_input(gamma_tensor);
  protect_input(beta_tensor);

  const int removed_count = FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_node_names, protected_node_names);

  VLOG(1) << "FuseLayerNormV2: Replaced '" << original_name
          << "' with MusaLayerNorm (removed_nodes=" << removed_count << ")";
  LOG(INFO) << "[FuseLayerNormV2][Apply] done apply for output="
            << original_name << ", removed_nodes=" << removed_count;

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaFuseLayerNormV2Fusion);
REGISTER_FUSION_KERNEL(MusaFuseLayerNormV2Fusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
