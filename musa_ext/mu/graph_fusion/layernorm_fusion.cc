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

#include <cmath>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Epsilon value for LayerNorm
constexpr float kDefaultEpsilon = 1e-6f;

// Helper to check if node has specific op type
bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

// Helper to find node's input producer
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;
  
  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  size_t colon_pos = node_name.find(':');
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

FusionMatchResult MusaLayerNormFusion::Match(const GraphDef& graph, int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }
  
  const NodeDef& start_node = graph.node(start_node_idx);
  if (HasOriginalSuffix(start_node.name())) {
    return FusionMatchResult{};
  }
  
  // Strategy: Look for AddV2/Add node and trace backwards
  if (IsOp(start_node, "AddV2") || IsOp(start_node, "Add")) {
    return MatchFromAddNode(graph, start_node_idx);
  }
  
  return FusionMatchResult{};
}

FusionMatchResult MusaLayerNormFusion::MatchFromAddNode(const GraphDef& graph, 
                                                         int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& add_node = graph.node(add_node_idx);
  
  if (!IsOp(add_node, "AddV2") && !IsOp(add_node, "Add")) {
    return result;
  }
  
  // Find the Mul input (gamma * normalized) and beta input
  const NodeDef* mul_node = nullptr;
  const NodeDef* beta_node = nullptr;
  
  for (int i = 0; i < add_node.input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, add_node.input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Mul")) {
      mul_node = input_node;
    } else if (IsOp(*input_node, "Const") || IsOp(*input_node, "VariableV2") ||
               IsOp(*input_node, "VarHandleOp")) {
      // This is likely beta
      beta_node = input_node;
    }
  }
  
  if (!mul_node) {
    return result;
  }
  
  // Trace back from Mul to find gamma and RealDiv/Mul
  const NodeDef* div_node = nullptr;
  const NodeDef* gamma_node = nullptr;
  
  for (int i = 0; i < mul_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, mul_node->input(i));
    if (!input_node) continue;
    
    // Accept RealDiv, Div, or Mul (TF constant folding may convert div to mul)
    if (IsOp(*input_node, "RealDiv") || IsOp(*input_node, "Div") || 
        IsOp(*input_node, "Mul")) {
      div_node = input_node;
    } else if (IsOp(*input_node, "Const") || IsOp(*input_node, "VariableV2") ||
               IsOp(*input_node, "VarHandleOp")) {
      // This is likely gamma
      gamma_node = input_node;
    }
  }
  
  if (!div_node) {
    return result;
  }
  
  // Trace back from RealDiv/Mul to find Sub
  const NodeDef* sub_node = nullptr;
  
  for (int i = 0; i < div_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, div_node->input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Sub")) {
      sub_node = input_node;
      break;
    }
  }
  
  if (!sub_node) {
    return result;
  }
  
  // Trace back from Sub to find Mean
  const NodeDef* mean_node = nullptr;
  
  for (int i = 0; i < sub_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, sub_node->input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Mean")) {
      mean_node = input_node;
      break;
    }
  }
  
  if (!mean_node) {
    return result;
  }
  
  // We found a valid LayerNorm chain!
  result.matched = true;
  result.matched_nodes.push_back(&add_node);
  result.matched_nodes.push_back(mul_node);
  result.matched_nodes.push_back(div_node);
  result.matched_nodes.push_back(sub_node);
  result.matched_nodes.push_back(mean_node);
  
  result.captured_nodes["output"] = &add_node;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["div"] = div_node;
  result.captured_nodes["sub"] = sub_node;
  result.captured_nodes["mean"] = mean_node;
  
  // Capture gamma and beta if found
  if (gamma_node) {
    result.captured_nodes["gamma"] = gamma_node;
    VLOG(2) << "MusaLayerNorm: Captured gamma: " << gamma_node->name();
  }
  if (beta_node) {
    result.captured_nodes["beta"] = beta_node;
    VLOG(2) << "MusaLayerNorm: Captured beta: " << beta_node->name();
  }
  
  // Find original input
  for (int i = 0; i < sub_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, sub_node->input(i));
    if (input_node && input_node != mean_node) {
      result.captured_nodes["input"] = input_node;
      break;
    }
  }
  
  result.captured_attrs["epsilon"] = std::to_string(kDefaultEpsilon);
  
  return result;
}

Status MusaLayerNormFusion::Apply(GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid LayerNorm match result");
  }
  
  if (!IsKernelAvailable()) {
    return Status::OK();
  }
  
  // Get captured nodes
  auto output_it = match_result.captured_nodes.find("output");
  auto input_it = match_result.captured_nodes.find("input");
  
  if (output_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT, "Missing output node in LayerNorm pattern");
  }
  
  const NodeDef* output_node = output_it->second;
  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";
  
  // Get gamma and beta nodes if available
  const NodeDef* gamma_node = nullptr;
  const NodeDef* beta_node = nullptr;
  
  auto gamma_it = match_result.captured_nodes.find("gamma");
  if (gamma_it != match_result.captured_nodes.end()) {
    gamma_node = gamma_it->second;
    VLOG(2) << "MusaLayerNorm: Apply found gamma: " << gamma_node->name();
  } else {
    VLOG(2) << "MusaLayerNorm: Apply did NOT find gamma";
  }
  
  auto beta_it = match_result.captured_nodes.find("beta");
  if (beta_it != match_result.captured_nodes.end()) {
    beta_node = beta_it->second;
    VLOG(2) << "MusaLayerNorm: Apply found beta: " << beta_node->name();
  } else {
    VLOG(2) << "MusaLayerNorm: Apply did NOT find beta";
  }
  
  // Get epsilon value
  float epsilon = kDefaultEpsilon;
  auto epsilon_it = match_result.captured_attrs.find("epsilon");
  if (epsilon_it != match_result.captured_attrs.end()) {
    epsilon = std::stof(epsilon_it->second);
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

  // Check if there's already a MusaLayerNorm node with the base name
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaLayerNorm") {
      VLOG(1) << "MusaLayerNorm: Output node " << original_name
              << " is already a fused node, skipping";
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
  const bool has_output_dtype =
      dtype_it != original_output_node->attr().end();
  if (has_output_dtype) {
    output_dtype = dtype_it->second;
  }

  std::string fused_input_name = input_name;
  if (fused_input_name.empty()) {
    auto mean_it = match_result.captured_nodes.find("mean");
    if (mean_it != match_result.captured_nodes.end() && mean_it->second &&
        mean_it->second->input_size() > 0) {
      fused_input_name = mean_it->second->input(0);
    } else {
      return Status(error::INVALID_ARGUMENT, "Cannot determine LayerNorm input");
    }
  }
  const std::string gamma_name = gamma_node ? gamma_node->name() : fused_input_name;
  const std::string beta_name = beta_node ? beta_node->name() : fused_input_name;
  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaLayerNorm");
  fused_node->set_device(output_device);
  
  // Set inputs: x, gamma, beta
  fused_node->add_input(fused_input_name);
  
  // Gamma input
  fused_node->add_input(gamma_name);
  
  // Beta input
  fused_node->add_input(beta_name);
  
  // Set attributes
  auto* attr = fused_node->mutable_attr();
  
  if (has_output_dtype) {
    (*attr)["T"] = output_dtype;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }
  
  (*attr)["epsilon"].set_f(epsilon);

  std::unordered_set<std::string> protected_node_names = {original_name};
  if (!input_name.empty()) {
    protected_node_names.insert(input_name);
  }
  if (!gamma_name.empty()) {
    protected_node_names.insert(gamma_name);
  }
  if (!beta_name.empty()) {
    protected_node_names.insert(beta_name);
  }

  const int removed_count = FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_node_names, protected_node_names);

  VLOG(1) << "MusaLayerNorm: Replaced '" << original_name
          << "' with MusaLayerNorm (removed_nodes=" << removed_count << ")";
  
  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaLayerNormFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaLayerNormFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
