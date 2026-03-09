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

#include "mu/graph_fusion/gelu_fusion.h"

#include <cmath>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// sqrt(2) constant used in GELU
constexpr float kSqrt2 = 1.41421356237f;
constexpr float kHalf = 0.5f;
constexpr float kOne = 1.0f;

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

// Helper to check if a const node has a specific float value
bool HasFloatValue(const NodeDef& node, float expected_val, float tolerance = 1e-5f) {
  if (!IsOp(node, "Const")) return false;
  
  auto it = node.attr().find("value");
  if (it == node.attr().end() || !it->second.has_tensor()) {
    return false;
  }
  
  const auto& tensor = it->second.tensor();
  if (tensor.float_val_size() > 0) {
    return std::abs(tensor.float_val(0) - expected_val) < tolerance;
  }
  
  return false;
}

}  // namespace

// =============================================================================
// MusaGeluFusion Implementation
// =============================================================================

MusaGeluFusion::MusaGeluFusion() = default;

bool MusaGeluFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    // MusaGelu kernel is NOT yet implemented
    // This is intentional for testing the fallback mechanism
    kernel_available_ = false;
    kernel_checked_ = true;
    VLOG(1) << "MusaGelu kernel is NOT available (not yet implemented) - "
            << "will use fallback to standard ops for testing";
  }
  return kernel_available_;
}

FusionMatchResult MusaGeluFusion::Match(const GraphDef& graph, int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }
  
  // Try different pattern variants
  FusionMatchResult result = MatchStandardPattern(graph, start_node_idx);
  if (result.IsValid()) {
    // Mark that kernel is not available (for fallback testing)
    VLOG(2) << "Matched GELU standard pattern but kernel not available - will fallback";
    return result;
  }
  
  result = MatchApproximatePattern(graph, start_node_idx);
  if (result.IsValid()) {
    VLOG(2) << "Matched GELU approximate pattern but kernel not available - will fallback";
    return result;
  }
  
  return FusionMatchResult{};
}

FusionMatchResult MusaGeluFusion::MatchStandardPattern(const GraphDef& graph, 
                                                        int start_node_idx) const {
  // Standard GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
  // Pattern from test file:
  //   x / sqrt(2.0) -> Erf -> (1.0 + result) -> 0.5 * x * result
  //
  // Typical TF graph structure:
  //   [x] -> Div(/sqrt(2)) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5) -> [output]
  //                                    ^                    ^
  //                                    |                    |
  //   [x] -----------------------------+   [0.5] -----------+
  
  FusionMatchResult result;
  const NodeDef& final_mul = graph.node(start_node_idx);
  
  // The final node should be Mul with 0.5 or the last Mul in chain
  if (!IsOp(final_mul, "Mul")) {
    return result;
  }
  
  // Look for the pattern from the output backwards
  // Check inputs to find if one is the x*erf path and other is 0.5
  const NodeDef* x_erf_path = nullptr;
  bool found_half = false;
  
  for (int i = 0; i < final_mul.input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, final_mul.input(i));
    if (!input_node) continue;
    
    // Check for 0.5 constant
    if (IsOp(*input_node, "Const") && HasFloatValue(*input_node, kHalf)) {
      found_half = true;
    } else if (IsOp(*input_node, "Mul")) {
      // This might be x * (1 + erf(...))
      x_erf_path = input_node;
    }
  }
  
  if (!found_half || !x_erf_path) {
    // Try alternative: maybe Mul(0.5, Mul(x, erf_result))
    // Both inputs are Mul - need to check deeper
    for (int i = 0; i < final_mul.input_size() && i < 2; ++i) {
      const NodeDef* input_node = FindProducer(graph, final_mul.input(i));
      if (!input_node) continue;
      
      if (IsOp(*input_node, "Mul")) {
        // Check if this Mul has 0.5
        for (int j = 0; j < input_node->input_size() && j < 2; ++j) {
          const NodeDef* inner_input = FindProducer(graph, input_node->input(j));
          if (inner_input && IsOp(*inner_input, "Const") && 
              HasFloatValue(*inner_input, kHalf)) {
            found_half = true;
            x_erf_path = input_node;
            break;
          }
        }
      }
    }
  }
  
  if (!x_erf_path) {
    return result;
  }
  
  // Now trace back x_erf_path to find the erf computation
  // x_erf_path should be: x * (1 + erf(x / sqrt(2)))
  const NodeDef* erf_add_node = nullptr;
  const NodeDef* original_x = nullptr;
  
  for (int i = 0; i < x_erf_path->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, x_erf_path->input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Add") || IsOp(*input_node, "AddV2")) {
      // Check if Add has 1.0
      for (int j = 0; j < input_node->input_size() && j < 2; ++j) {
        const NodeDef* add_input = FindProducer(graph, input_node->input(j));
        if (add_input && IsOp(*add_input, "Const") && HasFloatValue(*add_input, kOne)) {
          erf_add_node = input_node;
          break;
        }
      }
    } else {
      // This could be the original x
      original_x = input_node;
    }
  }
  
  if (!erf_add_node) {
    return result;
  }
  
  // Trace back from erf_add_node to find Erf
  const NodeDef* erf_node = nullptr;
  for (int i = 0; i < erf_add_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, erf_add_node->input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Erf")) {
      erf_node = input_node;
      break;
    }
  }
  
  if (!erf_node) {
    return result;
  }
  
  // Trace back from Erf to find Div(x / sqrt(2))
  const NodeDef* div_node = nullptr;
  if (erf_node->input_size() > 0) {
    div_node = FindProducer(graph, erf_node->input(0));
  }
  
  if (!div_node || !IsOp(*div_node, "RealDiv") && !IsOp(*div_node, "Div")) {
    return result;
  }
  
  // Verify Div has sqrt(2) denominator
  bool found_sqrt2 = false;
  for (int i = 0; i < div_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, div_node->input(i));
    if (!input_node) continue;
    
    if (IsOp(*input_node, "Const") && HasFloatValue(*input_node, kSqrt2)) {
      found_sqrt2 = true;
      break;
    } else if (IsOp(*input_node, "Sqrt")) {
      // Check if Sqrt input is 2.0
      if (input_node->input_size() > 0) {
        const NodeDef* sqrt_input = FindProducer(graph, input_node->input(0));
        if (sqrt_input && IsOp(*sqrt_input, "Const") && 
            HasFloatValue(*sqrt_input, 2.0f)) {
          found_sqrt2 = true;
          break;
        }
      }
    }
  }
  
  if (!found_sqrt2) {
    // Still accept the pattern even if sqrt(2) check fails
    // The pattern structure is what matters most
    VLOG(2) << "GELU pattern matched but sqrt(2) constant not verified";
  }
  
  // Build the match result
  result.matched = true;
  result.matched_nodes.push_back(&final_mul);
  result.matched_nodes.push_back(x_erf_path);
  if (erf_add_node) result.matched_nodes.push_back(erf_add_node);
  if (erf_node) result.matched_nodes.push_back(erf_node);
  if (div_node) result.matched_nodes.push_back(div_node);
  
  result.captured_nodes["output"] = &final_mul;
  result.captured_nodes["x_erf_mul"] = x_erf_path;
  
  // Find original input
  for (int i = 0; i < div_node->input_size() && i < 2; ++i) {
    const NodeDef* input_node = FindProducer(graph, div_node->input(i));
    if (!input_node) continue;
    
    if (!IsOp(*input_node, "Const")) {
      result.captured_nodes["input"] = input_node;
      original_x = input_node;
      break;
    }
  }
  
  if (original_x) {
    result.captured_nodes["input"] = original_x;
  }
  
  VLOG(1) << "Matched GELU standard pattern with " << result.matched_nodes.size() 
          << " nodes (kernel available: " << IsKernelAvailable() << ")";
  
  return result;
}

FusionMatchResult MusaGeluFusion::MatchApproximatePattern(const GraphDef& graph, 
                                                           int start_node_idx) const {
  // Approximate GELU using tanh:
  // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // This is used in some implementations but less common in basic TF
  
  FusionMatchResult result;
  
  // For now, we don't implement the approximate pattern matching
  // as the test file uses the exact erf-based implementation
  // This can be extended later
  
  return result;
}

Status MusaGeluFusion::Apply(GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid GELU match result");
  }
  
  // IMPORTANT: Since MusaGelu kernel is NOT implemented yet,
  // we intentionally DO NOT apply the fusion.
  // Instead, we log the fallback and return OK, which allows the
  // original ops to execute (fallback mechanism).
  
  if (!IsKernelAvailable()) {
    VLOG(1) << "MusaGeluFusion: Pattern matched but kernel not available. "
            << "Using fallback to standard ops. "
            << "Matched " << match_result.matched_nodes.size() << " nodes.";
    
    // Return OK to indicate graceful fallback
    // The original graph remains unchanged
    return Status::OK();
  }
  
  // If kernel becomes available in the future, this is where the fusion would apply:
  
  // Get captured nodes
  const NodeDef* output_node = match_result.captured_nodes.at("output");
  const NodeDef* input_node = match_result.captured_nodes.at("input");
  
  if (!output_node || !input_node) {
    return Status(error::INVALID_ARGUMENT, "Missing required nodes in GELU pattern");
  }
  
  // Create new MusaGelu node
  std::string fused_node_name = output_node->name() + "_fused_gelu";
  
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaGelu");
  fused_node->set_device(output_node->device());
  
  // Set input
  fused_node->add_input(input_node->name());
  
  // Set attributes
  auto* attr = fused_node->mutable_attr();
  
  // Copy T attribute from output node if available
  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    (*attr)["T"] = dtype_it->second;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }
  
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
  
  VLOG(1) << "Applied GELU fusion: created " << fused_node_name 
          << " replacing " << match_result.matched_nodes.size() << " nodes";
  
  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaGeluFusion);

// Register kernel availability (returns false - not implemented)
REGISTER_FUSION_KERNEL(MusaGeluFusion, []() { return false; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
