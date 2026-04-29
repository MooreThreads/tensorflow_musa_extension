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

#include "kernels/fusion/tensordot_bias_fusion.h"

#include <unordered_set>

#include "mu/optimizer/graph_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Weight node valid types
const std::unordered_set<std::string> kWeightOps = {
    "Identity", "ReadVariableOp", "Const", "Reshape"};

// Helper to check if node has specific op type
bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

// Helper to check if node is a valid weight node
bool IsWeightOp(const NodeDef& node) {
  return kWeightOps.find(node.op()) != kWeightOps.end();
}

// Helper to find node's input producer
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  std::string node_name = input;
  // Handle control dependencies
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  // Handle output port suffix
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

// Helper to get clean node name (remove port suffix and control dep prefix)
std::string GetCleanName(const std::string& input) {
  if (input.empty()) return "";

  std::string name = input;
  if (name[0] == '^') {
    name = name.substr(1);
  }
  size_t colon_pos = name.find(':');
  if (colon_pos != std::string::npos) {
    name = name.substr(0, colon_pos);
  }
  return name;
}

}  // namespace

// =============================================================================
// MusaTensorDotBiasFusion Implementation
// =============================================================================
//
// This fusion pattern matches AFTER MusaTensorDotFusion has already run.
// Pattern:
//   MusaTensorDot (already fused from tensordot_fusion)
//       ↓
//   BiasAdd
//       ↓
//   (consumers)
//
// We fuse these two nodes into a single MusaTensorDotBias op.
// =============================================================================

MusaTensorDotBiasFusion::MusaTensorDotBiasFusion() = default;

bool MusaTensorDotBiasFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaTensorDotBiasFusion::Match(const GraphDef& graph,
                                                 int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    VLOG(2) << "[TensorDotBias::Match] RETURN empty: node_idx out of range";
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  // Entry point is BiasAdd node
  if (!IsOp(start_node, "BiasAdd")) {
    return FusionMatchResult{};
  }

  return MatchFromBiasAddNode(graph, start_node_idx);
}

FusionMatchResult MusaTensorDotBiasFusion::MatchFromBiasAddNode(
    const GraphDef& graph, int bias_add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& bias_add = graph.node(bias_add_node_idx);

  VLOG(2) << "[TensorDotBias::Match] MatchFromBiasAddNode ENTER, node="
          << bias_add.name();

  // =========================================================================
  // BiasAdd must have 2 inputs:
  //   - input[0]: MusaTensorDot output (already fused)
  //   - input[1]: bias weights (ReadVariableOp/Identity/etc)
  // =========================================================================
  if (bias_add.input_size() != 2) {
    VLOG(2) << "[TensorDotBias::Match] FAIL: BiasAdd input_size="
            << bias_add.input_size() << " (need 2), node=" << bias_add.name();
    return result;
  }

  const NodeDef* tensordot_output = FindProducer(graph, bias_add.input(0));
  const NodeDef* bias_weights = FindProducer(graph, bias_add.input(1));

  // Check input[0] is MusaTensorDot (already fused by tensordot_fusion)
  if (!tensordot_output || !IsOp(*tensordot_output, "MusaTensorDot")) {
    VLOG(2)
        << "[TensorDotBias::Match] FAIL: input[0] is not MusaTensorDot, actual="
        << (tensordot_output ? tensordot_output->op() : "NULL")
        << ", node=" << bias_add.name();
    return result;
  }

  // Check input[1] is a valid weight node
  if (!bias_weights || !IsWeightOp(*bias_weights)) {
    VLOG(2)
        << "[TensorDotBias::Match] FAIL: input[1] is not weight node, actual="
        << (bias_weights ? bias_weights->op() : "NULL")
        << ", node=" << bias_add.name();
    return result;
  }

  VLOG(2) << "[TensorDotBias::Match] PASS initial check: BiasAdd="
          << bias_add.name() << ", MusaTensorDot=" << tensordot_output->name()
          << ", bias_weights=" << bias_weights->name();

  // =========================================================================
  // Extract axes from MusaTensorDot node
  // =========================================================================
  std::vector<int> axes_a, axes_b;
  auto axes_a_it = tensordot_output->attr().find("axes_a");
  if (axes_a_it != tensordot_output->attr().end()) {
    const AttrValue& av = axes_a_it->second;
    for (int i = 0; i < av.list().i_size(); ++i) {
      axes_a.push_back(static_cast<int>(av.list().i(i)));
    }
  }
  if (axes_a.empty()) {
    axes_a.push_back(-1);
    LOG(WARNING) << "[TensorDotBias::Match] could not extract axes_a from "
                 << "MusaTensorDot, using default [-1], node="
                 << tensordot_output->name();
  }

  auto axes_b_it = tensordot_output->attr().find("axes_b");
  if (axes_b_it != tensordot_output->attr().end()) {
    const AttrValue& av = axes_b_it->second;
    for (int i = 0; i < av.list().i_size(); ++i) {
      axes_b.push_back(static_cast<int>(av.list().i(i)));
    }
  }
  if (axes_b.empty()) {
    axes_b.push_back(0);
  }

  VLOG(2) << "[TensorDotBias::Match] extracted axes_a=" <<
      [&]() {
        std::string s;
        for (size_t i = 0; i < axes_a.size(); ++i) {
          if (i > 0) s += ",";
          s += std::to_string(axes_a[i]);
        }
        return s;
      }()
          << ", axes_b=" << [&]() {
               std::string s;
               for (size_t i = 0; i < axes_b.size(); ++i) {
                 if (i > 0) s += ",";
                 s += std::to_string(axes_b[i]);
               }
               return s;
             }();

  // =========================================================================
  // Build match result
  // =========================================================================
  result.matched = true;

  // Add matched nodes (order: consumer first, then producer)
  result.matched_nodes.push_back(&bias_add);
  result.matched_nodes.push_back(tensordot_output);

  // Capture critical nodes
  result.captured_nodes["bias_add"] = &bias_add;
  result.captured_nodes["tensordot"] = tensordot_output;
  result.captured_nodes["bias_weights"] = bias_weights;

  // Capture attributes needed for Apply
  result.captured_attrs["original_input"] = tensordot_output->input(0);
  result.captured_attrs["tensordot_weight_input"] = tensordot_output->input(1);
  // Use BiasAdd.input(1) instead of bias_weights->name() to preserve port
  // information
  result.captured_attrs["bias_weights_input"] = bias_add.input(1);

  // Serialize axes
  std::string axes_a_str, axes_b_str;
  for (size_t i = 0; i < axes_a.size(); ++i) {
    if (i > 0) axes_a_str += ",";
    axes_a_str += std::to_string(axes_a[i]);
  }
  for (size_t i = 0; i < axes_b.size(); ++i) {
    if (i > 0) axes_b_str += ",";
    axes_b_str += std::to_string(axes_b[i]);
  }
  result.captured_attrs["axes_a"] = axes_a_str;
  result.captured_attrs["axes_b"] = axes_b_str;

  VLOG(1) << "[TensorDotBias::Match] SUCCESS matched=" << bias_add.name()
          << ", tensordot=" << tensordot_output->name()
          << ", bias_weights=" << bias_weights->name();

  return result;
}

Status MusaTensorDotBiasFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  VLOG(2) << "[TensorDotBias::Apply] ENTER, matched=" << match_result.matched
          << ", nodes_count=" << match_result.matched_nodes.size()
          << ", kernel_available=" << IsKernelAvailable();

  if (!match_result.IsValid()) {
    VLOG(2) << "[TensorDotBias::Apply] RETURN: invalid match result";
    return Status(error::INVALID_ARGUMENT,
                  "Invalid TensorDotBias match result");
  }

  if (!IsKernelAvailable()) {
    VLOG(2) << "[TensorDotBias::Apply] RETURN: kernel not available, skipping "
               "fusion";
    return Status::OK();
  }

  // Get critical nodes
  auto bias_add_it = match_result.captured_nodes.find("bias_add");
  auto tensordot_it = match_result.captured_nodes.find("tensordot");
  auto bias_weights_it = match_result.captured_nodes.find("bias_weights");

  if (bias_add_it == match_result.captured_nodes.end()) {
    VLOG(2) << "[TensorDotBias::Apply] RETURN: missing bias_add node in "
               "captured_nodes";
    return Status(error::INVALID_ARGUMENT,
                  "Missing bias_add node in TensorDotBias pattern");
  }

  const NodeDef* bias_add_node = bias_add_it->second;
  std::string output_name = bias_add_node->name();
  std::string output_device = bias_add_node->device();
  VLOG(2) << "[TensorDotBias::Apply] bias_add_node=" << output_name;

  // Check if already fused
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaTensorDotBias") {
      VLOG(2) << "[TensorDotBias::Apply] RETURN: already fused, node="
              << output_name;
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  // Get input names from captured attrs
  std::string input_a_name;
  std::string tensordot_weight_name;
  std::string bias_weights_name;

  auto original_input_it = match_result.captured_attrs.find("original_input");
  if (original_input_it != match_result.captured_attrs.end() &&
      !original_input_it->second.empty()) {
    input_a_name = original_input_it->second;
  } else {
    VLOG(2) << "[TensorDotBias::Apply] RETURN: cannot determine input A";
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine TensorDotBias input A");
  }

  auto tensordot_weight_input_it =
      match_result.captured_attrs.find("tensordot_weight_input");
  if (tensordot_weight_input_it != match_result.captured_attrs.end() &&
      !tensordot_weight_input_it->second.empty()) {
    tensordot_weight_name = tensordot_weight_input_it->second;
  } else {
    VLOG(2)
        << "[TensorDotBias::Apply] RETURN: cannot determine tensordot weight";
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine TensorDotBias tensordot weight");
  }

  auto bias_weights_input_it =
      match_result.captured_attrs.find("bias_weights_input");
  if (bias_weights_input_it != match_result.captured_attrs.end() &&
      !bias_weights_input_it->second.empty()) {
    bias_weights_name = bias_weights_input_it->second;
  } else if (bias_weights_it != match_result.captured_nodes.end() &&
             bias_weights_it->second) {
    bias_weights_name = bias_weights_it->second->name();
  } else {
    VLOG(2) << "[TensorDotBias::Apply] RETURN: cannot determine bias weights";
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine TensorDotBias bias weights");
  }

  // Get data type from BiasAdd
  DataType dtype = DT_FLOAT;
  auto dtype_it = bias_add_node->attr().find("T");
  if (dtype_it != bias_add_node->attr().end()) {
    dtype = dtype_it->second.type();
  }

  // Extract axes from captured attrs
  std::vector<int> axes_a, axes_b;
  auto axes_a_it = match_result.captured_attrs.find("axes_a");
  if (axes_a_it != match_result.captured_attrs.end()) {
    const std::string& s = axes_a_it->second;
    if (!s.empty()) {
      size_t start = 0;
      size_t end;
      while ((end = s.find(',', start)) != std::string::npos) {
        axes_a.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
      }
      axes_a.push_back(std::stoi(s.substr(start)));
    }
  }
  if (axes_a.empty()) axes_a.push_back(-1);

  auto axes_b_it = match_result.captured_attrs.find("axes_b");
  if (axes_b_it != match_result.captured_attrs.end()) {
    const std::string& s = axes_b_it->second;
    if (!s.empty()) {
      size_t start = 0;
      size_t end;
      while ((end = s.find(',', start)) != std::string::npos) {
        axes_b.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
      }
      axes_b.push_back(std::stoi(s.substr(start)));
    }
  }
  if (axes_b.empty()) axes_b.push_back(0);

  VLOG(2) << "[TensorDotBias::Apply] input_a=" << input_a_name
          << ", tensordot_weight=" << tensordot_weight_name
          << ", bias_weights=" << bias_weights_name;

  // =========================================================================
  // Nodes to remove: MusaTensorDot and BiasAdd
  // Keep bias_weights as it's an external input
  // =========================================================================
  std::unordered_set<std::string> nodes_to_remove;
  nodes_to_remove.insert(output_name);  // BiasAdd will be replaced

  auto tensordot_node_it = match_result.captured_nodes.find("tensordot");
  if (tensordot_node_it != match_result.captured_nodes.end() &&
      tensordot_node_it->second) {
    nodes_to_remove.insert(tensordot_node_it->second->name());
  }

  // Check if any external node depends on the intermediate MusaTensorDot output
  // (should not happen since BiasAdd consumes it, but be safe)
  std::string tensordot_output_name;
  if (tensordot_node_it != match_result.captured_nodes.end() &&
      tensordot_node_it->second) {
    tensordot_output_name = tensordot_node_it->second->name();

    for (int i = 0; i < graph->node_size(); ++i) {
      const NodeDef& node = graph->node(i);
      if (nodes_to_remove.count(node.name())) continue;

      for (int j = 0; j < node.input_size(); ++j) {
        std::string producer = GetCleanName(node.input(j));
        if (producer == tensordot_output_name && producer != output_name) {
          // External node depends on tensordot output, keep it
          VLOG(2) << "[TensorDotBias::Apply] keeping tensordot due to external "
                     "dep from "
                  << node.name();
          nodes_to_remove.erase(tensordot_output_name);
          break;
        }
      }
    }
  }

  VLOG(2) << "[TensorDotBias::Apply] will remove " << nodes_to_remove.size()
          << " nodes";

  // =========================================================================
  // Remove fused nodes (in reverse topological order)
  // =========================================================================
  int removed_count = 0;

  // First remove BiasAdd (will be replaced by fused node)
  int bias_add_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (bias_add_idx >= 0) {
    FusionGraphUtils::RemoveNode(graph, bias_add_idx);
    removed_count++;
  }

  // Then remove MusaTensorDot if not kept
  if (nodes_to_remove.count(tensordot_output_name)) {
    int tensordot_idx =
        FusionGraphUtils::FindNodeIndex(*graph, tensordot_output_name);
    if (tensordot_idx >= 0) {
      FusionGraphUtils::RemoveNode(graph, tensordot_idx);
      removed_count++;
    }
  }

  VLOG(2) << "[TensorDotBias::Apply] removed " << removed_count << " nodes";

  // =========================================================================
  // Create fused node
  // =========================================================================
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(output_name);
  fused_node->set_op("MusaTensorDotBias");
  fused_node->set_device(output_device);

  // Inputs: original_input, tensordot_weight, bias_weights
  fused_node->add_input(input_a_name);
  fused_node->add_input(tensordot_weight_name);
  fused_node->add_input(bias_weights_name);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"].set_type(dtype);

  auto* axes_a_list = (*attr)["axes_a"].mutable_list();
  for (int a : axes_a) axes_a_list->add_i(a);

  auto* axes_b_list = (*attr)["axes_b"].mutable_list();
  for (int b : axes_b) axes_b_list->add_i(b);

  VLOG(1) << "[TensorDotBias::Apply] SUCCESS fused to " << output_name
          << ", removed=" << removed_count
          << ", graph_nodes=" << graph->node_size();

  return Status::OK();
}

// Register fusion pattern
REGISTER_FUSION_PATTERN(MusaTensorDotBiasFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaTensorDotBiasFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
