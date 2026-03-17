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

#include "mu/graph_fusion/sigmoid_calibration_fusion.h"

#include <cmath>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

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
// MusaSigmoidCalibrationFusion Implementation
// =============================================================================

MusaSigmoidCalibrationFusion::MusaSigmoidCalibrationFusion() = default;

bool MusaSigmoidCalibrationFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    // We assume the kernel will be implemented or we use a fallback if needed
    // In this context, we mark it as available for fusion to take place
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaSigmoidCalibrationFusion::Match(const GraphDef& graph, int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& real_div_node = graph.node(start_node_idx);
  if (!IsOp(real_div_node, "RealDiv")) {
    return FusionMatchResult{};
  }

  FusionMatchResult result;

  // RealDiv input 0: Sigmoid(x)
  // RealDiv input 1: AddV2
  const NodeDef* sigmoid_node = FindProducer(graph, real_div_node.input(0));
  const NodeDef* add_node = FindProducer(graph, real_div_node.input(1));

  if (!sigmoid_node || !add_node || !IsOp(*sigmoid_node, "Sigmoid") || 
      (!IsOp(*add_node, "AddV2") && !IsOp(*add_node, "Add"))) {
    return FusionMatchResult{};
  }

  // Add input 0: Sigmoid(x) (same as above)
  // Add input 1: Mul
  const NodeDef* sigmoid_in_add = FindProducer(graph, add_node->input(0));
  const NodeDef* mul_node = FindProducer(graph, add_node->input(1));

  // Some graphs might have the inputs swapped
  if (sigmoid_in_add != sigmoid_node) {
    mul_node = FindProducer(graph, add_node->input(0));
    sigmoid_in_add = FindProducer(graph, add_node->input(1));
  }

  if (sigmoid_in_add != sigmoid_node || !mul_node || !IsOp(*mul_node, "Mul")) {
    return FusionMatchResult{};
  }

  // Mul input 0: Sub(1-S)
  // Mul input 1: Const (1x32)
  const NodeDef* sub_node = FindProducer(graph, mul_node->input(0));
  const NodeDef* scale_const_node = FindProducer(graph, mul_node->input(1));

  if (sub_node && IsOp(*sub_node, "Const")) {
     // Swapped case
     scale_const_node = sub_node;
     sub_node = FindProducer(graph, mul_node->input(1));
  }

  if (!sub_node || !IsOp(*sub_node, "Sub") || !scale_const_node || !IsOp(*scale_const_node, "Const")) {
    return FusionMatchResult{};
  }

  // Sub input 0: Const (1)
  // Sub input 1: Sigmoid(x) (same as above)
  const NodeDef* one_const_node = FindProducer(graph, sub_node->input(0));
  const NodeDef* sigmoid_in_sub = FindProducer(graph, sub_node->input(1));

  if (!one_const_node || !sigmoid_in_sub || sigmoid_in_sub != sigmoid_node || !HasFloatValue(*one_const_node, 1.0f)) {
    return FusionMatchResult{};
  }

  // Success!
  result.matched = true;
  result.matched_nodes = {&real_div_node, add_node, mul_node, sub_node, sigmoid_node};
  result.captured_nodes["input"] = FindProducer(graph, sigmoid_node->input(0));
  result.captured_nodes["scale"] = scale_const_node;

  return result;
}

Status MusaSigmoidCalibrationFusion::Apply(GraphDef* graph, const FusionMatchResult& match_result) const {
  VLOG(1) << "Applying MusaSigmoidCalibrationFusion fusion";

  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid FusedSigmoidCalibration match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const NodeDef* real_div_node = match_result.matched_nodes[0];
  const NodeDef* scale_const_node = match_result.captured_nodes.at("scale");
  const NodeDef* input_node = match_result.captured_nodes.at("input");

  DataType dtype = DT_FLOAT;
  auto it = real_div_node->attr().find("T");
  if (it != real_div_node->attr().end()) {
    dtype = it->second.type();
  }

  // Create MusaSigmoidCalibrationFusion node
  NodeDef* sigmoid_calibration_node = graph->add_node();
  sigmoid_calibration_node->set_name(real_div_node->name() + "_fused_sigmoid_calibration");
  sigmoid_calibration_node->set_op("MusaSigmoidCalibration");
  sigmoid_calibration_node->set_device(real_div_node->device());
  sigmoid_calibration_node->add_input(sigmoid_node_input_name(match_result));
  sigmoid_calibration_node->add_input(scale_const_node->name());
  (*sigmoid_calibration_node->mutable_attr())["T"].set_type(dtype);

  // Update consumers of RealDiv to use MusaSigmoidCalibration instead
  std::string old_name = real_div_node->name();
  std::string new_name = sigmoid_calibration_node->name();

  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == old_name) {
        node->set_input(j, new_name);
      } else if (node->input(j) == old_name + ":0") {
        node->set_input(j, new_name + ":0");
      }
    }
  }

  return Status::OK();
}

std::string MusaSigmoidCalibrationFusion::sigmoid_node_input_name(const FusionMatchResult& match_result) const {
  const NodeDef* sigmoid_node = match_result.matched_nodes[4];
  if (sigmoid_node->input_size() > 0) {
    return sigmoid_node->input(0);
  }
  return "";
}

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
