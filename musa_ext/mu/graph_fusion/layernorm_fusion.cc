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
  
  // Create new MusaLayerNorm node
  std::string fused_node_name = output_node->name() + "_fused_layernorm";
  
  // Check if this output node has already been fused (avoid duplicates)
  // Extract the base name (remove trailing "_original" suffix if present)
  std::string base_name = output_node->name();
  if (base_name.size() > 9 && base_name.substr(base_name.size() - 9) == "_original") {
    base_name = base_name.substr(0, base_name.size() - 9);
  }
  
  // Check if there's already a MusaLayerNorm node with the base name
  for (const auto& node : graph->node()) {
    if (node.name() == base_name && node.op() == "MusaLayerNorm") {
      VLOG(1) << "MusaLayerNorm: Output node " << base_name 
              << " is already a fused node, skipping";
      return Status::OK();
    }
  }
  
  VLOG(1) << "MusaLayerNorm: Creating fused node: " << fused_node_name;
  
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaLayerNorm");
  fused_node->set_device(output_node->device());
  
  // Set inputs: x, gamma, beta
  if (input_it != match_result.captured_nodes.end() && input_it->second) {
    fused_node->add_input(input_it->second->name());
  } else {
    auto mean_it = match_result.captured_nodes.find("mean");
    if (mean_it != match_result.captured_nodes.end() && mean_it->second && 
        mean_it->second->input_size() > 0) {
      fused_node->add_input(mean_it->second->input(0));
    } else {
      return Status(error::INVALID_ARGUMENT, "Cannot determine LayerNorm input");
    }
  }
  
  // Gamma input
  if (gamma_node) {
    fused_node->add_input(gamma_node->name());
  } else {
    fused_node->add_input(fused_node->input(0));
  }
  
  // Beta input
  if (beta_node) {
    fused_node->add_input(beta_node->name());
  } else {
    fused_node->add_input(fused_node->input(0));
  }
  
  // Set attributes
  auto* attr = fused_node->mutable_attr();
  
  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    (*attr)["T"] = dtype_it->second;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }
  
  (*attr)["epsilon"].set_f(epsilon);
  
  // Redirect all inputs from the output node to the fused node
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == fused_node_name) continue;
    
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == output_node->name()) {
        node->set_input(j, fused_node_name);
      } else if (node->input(j).find(output_node->name() + ":") == 0) {
        std::string suffix = node->input(j).substr(output_node->name().length());
        node->set_input(j, fused_node_name + suffix);
      }
    }
  }
  
  // Rename the original output node and give the fused node the original name
  // This ensures that direct fetches of the output tensor get the fused result
  std::string original_name = output_node->name();
  const_cast<NodeDef*>(output_node)->set_name(original_name + "_original");
  fused_node->set_name(original_name);
  
  // Also update any references to the renamed original node
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == original_name) continue;  // Skip the fused node (now has original name)
    
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == original_name + "_original") {
        // These should point to the fused node (which now has the original name)
        node->set_input(j, original_name);
      }
    }
  }
  
  VLOG(1) << "MusaLayerNorm: Renamed fused node to " << original_name;
  
  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaLayerNormFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaLayerNormFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
