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

/// fusion pattern: PRelu(x, alpha) = max(0, x) + alpha * min(0, x)
///
/// Match pattern (from AddV2 backward):
///   AddV2
///   ├── Relu1 -> Select
///   └── Mul
///       ├── Neg1 -> Identity/Const (alpha)
///       └── Relu2 -> Neg2 -> Select (same as Relu1's input)
///
/// The fused op: MusaPRelu(x, alpha) -> output
#include "kernels/fusion/prelu_fusion.h"

#include "kernels/fusion/fusion_pattern_manager.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

bool IsAddOp(const NodeDef& node) {
  return IsOp(node, "Add") || IsOp(node, "AddV2");
}

std::string GetProducerName(const std::string& input) {
  if (input.empty()) return "";
  std::string name = input;
  if (name[0] == '^') name = name.substr(1);
  size_t colon_pos = name.find(':');
  if (colon_pos != std::string::npos) name = name.substr(0, colon_pos);
  return name;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  std::string node_name = GetProducerName(input);
  if (node_name.empty()) return nullptr;

  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return &graph.node(i);
    }
  }
  return nullptr;
}

DataType GetNodeDType(const NodeDef& node) {
  auto it = node.attr().find("T");
  return (it != node.attr().end()) ? it->second.type() : DT_INVALID;
}

int CountConsumers(const GraphDef& graph, const std::string& node_name) {
  int count = 0;
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& n = graph.node(i);
    for (int j = 0; j < n.input_size(); ++j) {
      if (GetProducerName(n.input(j)) == node_name) {
        count++;
      }
    }
  }
  return count;
}

// Check if node is Identity wrapping a Const, or directly a Const
const NodeDef* FindAlphaSource(const GraphDef& graph, const NodeDef* node) {
  if (!node) return nullptr;

  // Direct Const node
  if (IsOp(*node, "Const")) {
    return node;
  }

  // Identity wrapping a Const
  if (IsOp(*node, "Identity") && node->input_size() >= 1) {
    const NodeDef* producer = FindProducer(graph, node->input(0));
    if (producer && IsOp(*producer, "Const")) {
      return producer;
    }
  }

  return nullptr;
}

}  // namespace

// =============================================================================
// MusaPReluFusion Implementation
// =============================================================================

MusaPReluFusion::MusaPReluFusion() = default;

bool MusaPReluFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaPReluFusion::Match(const GraphDef& graph,
                                         int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  // Try to match starting from AddV2 node
  if (IsAddOp(start_node)) {
    return MatchFromAddV2Node(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaPReluFusion::MatchFromAddV2Node(
    const GraphDef& graph, int addv2_node_idx) const {
  FusionMatchResult result;
  const NodeDef& addv2_node = graph.node(addv2_node_idx);

  if (!IsAddOp(addv2_node) || addv2_node.input_size() < 2) {
    VLOG(2) << "MusaPReluFusion: Invalid AddV2 node";
    return result;
  }

  // Check if already fused
  const std::string& name = addv2_node.name();
  if (name.size() > 9 && name.substr(name.size() - 9) == "_original") {
    VLOG(2) << "MusaPRelu: Node '" << name << "' already fused, skipping";
    return result;
  }

  VLOG(2) << "MusaPReluFusion: Starting match from AddV2 node: " << name;

  // ===========================================================================
  // Step 1: Identify Relu1 and Mul branches from AddV2
  // ===========================================================================
  const NodeDef* relu1_node = nullptr;  // Positive part: Relu(Select)
  const NodeDef* mul_node =
      nullptr;  // Negative part: Mul(Neg(alpha), Relu(Neg(Select)))

  const NodeDef* input0 = FindProducer(graph, addv2_node.input(0));
  const NodeDef* input1 = FindProducer(graph, addv2_node.input(1));
  if (!input0 || !input1) {
    VLOG(2) << "MusaPReluFusion: Could not find producers for AddV2 inputs";
    return result;
  }
  // Look for: AddV2(Relu, Mul) pattern
  if (IsOp(*input0, "Relu") && IsOp(*input1, "Mul")) {
    relu1_node = input0;
    mul_node = input1;
    VLOG(2) << "MusaPReluFusion: Found Relu at input " << 0 << ", Mul at input "
            << 1;
  }

  if (!relu1_node || !mul_node) {
    VLOG(2) << "MusaPReluFusion: Could not find Relu+Mul inputs to AddV2";
    return result;
  }

  // ===========================================================================
  // Step 2: Relu1's input must be a Select node
  // ===========================================================================
  if (relu1_node->input_size() < 1) {
    VLOG(2) << "MusaPReluFusion: Relu1 has no input";
    return result;
  }

  const NodeDef* select_node = FindProducer(graph, relu1_node->input(0));

  VLOG(2) << "MusaPReluFusion: Found Select node: " << select_node->name();
  // ===========================================================================
  // Step 3: Parse Mul's inputs: Mul(Neg1, Relu2)
  // ===========================================================================
  if (mul_node->input_size() < 2) {
    VLOG(2) << "MusaPReluFusion: Mul node has insufficient inputs";
    return result;
  }

  const NodeDef* const_node = nullptr;  // Neg(alpha)
  const NodeDef* relu2_node = nullptr;  // Relu(Neg(Select))

  const NodeDef* mul_input0 = FindProducer(graph, mul_node->input(0));
  const NodeDef* mul_input1 = FindProducer(graph, mul_node->input(1));
  if (!mul_input0 || !mul_input1) {
    VLOG(2) << "MusaPReluFusion: Could not find producers for Mul inputs";
    return result;
  }

  if (IsOp(*mul_input0, "Const")) {
    const_node = mul_input0;
  }

  // Look for: Mul(Neg, Relu)
  if (IsOp(*mul_input1, "Relu")) {
    relu2_node = mul_input1;
  }

  if (!const_node || !relu2_node) {
    VLOG(2) << "MusaPReluFusion: Could not find Const+Relu inputs to Mul";
    return result;
  }

  // ===========================================================================
  // Step 5: Relu2's input must be Neg2
  // ===========================================================================
  if (relu2_node->input_size() < 1) {
    VLOG(2) << "MusaPReluFusion: Relu2 has no input";
    return result;
  }

  const NodeDef* neg2_node = FindProducer(graph, relu2_node->input(0));
  if (!neg2_node || !IsOp(*neg2_node, "Neg")) {
    VLOG(2) << "MusaPReluFusion: Relu2's input is not a Neg node";
    return result;
  }

  VLOG(2) << "MusaPReluFusion: Found Neg2: " << neg2_node->name();

  // ===========================================================================
  // Step 6: Neg2's input must be the SAME Select node as Relu1's input
  // ===========================================================================
  if (neg2_node->input_size() < 1) {
    VLOG(2) << "MusaPReluFusion: Neg2 has no input";
    return result;
  }

  const NodeDef* neg2_input = FindProducer(graph, neg2_node->input(0));
  if (neg2_input != select_node) {
    VLOG(2)
        << "MusaPReluFusion: Neg2's input is not the same Select as Relu1's. "
        << "Neg2 input: " << (neg2_input ? neg2_input->name() : "null")
        << ", Select: " << select_node->name();
    return result;
  }

  VLOG(2) << "MusaPReluFusion: Neg2 and Relu1 share the same Select node";

  // ===========================================================================
  // Step 7: Validate single-consumer for intermediate nodes
  // ===========================================================================

  // Relu1 must only be consumed by AddV2
  int relu1_consumers = CountConsumers(graph, relu1_node->name());
  if (relu1_consumers != 1) {
    VLOG(2) << "MusaPReluFusion: Relu1 has " << relu1_consumers
            << " consumers, expected 1";
    return result;
  }

  // Neg2 must only be consumed by Relu2
  int neg2_consumers = CountConsumers(graph, neg2_node->name());
  if (neg2_consumers != 1) {
    VLOG(2) << "MusaPReluFusion: Neg2 has " << neg2_consumers
            << " consumers, expected 1";
    return result;
  }

  // Relu2 must only be consumed by Mul
  int relu2_consumers = CountConsumers(graph, relu2_node->name());
  if (relu2_consumers != 1) {
    VLOG(2) << "MusaPReluFusion: Relu2 has " << relu2_consumers
            << " consumers, expected 1";
    return result;
  }

  // Const must only be consumed by Mul
  int const_consumers = CountConsumers(graph, const_node->name());
  if (const_consumers != 1) {
    VLOG(2) << "MusaPReluFusion: Const has " << const_consumers
            << " consumers, expected 1";
    return result;
  }

  // Mul must only be consumed by AddV2
  int mul_consumers = CountConsumers(graph, mul_node->name());
  if (mul_consumers != 1) {
    VLOG(2) << "MusaPReluFusion: Mul has " << mul_consumers
            << " consumers, expected 1";
    return result;
  }

  // ===========================================================================
  // Step 8: Validate dtype consistency
  // ===========================================================================
  DataType dt_select = GetNodeDType(*select_node);
  DataType dt_addv2 = GetNodeDType(addv2_node);

  if (dt_select != DT_INVALID && dt_addv2 != DT_INVALID &&
      dt_select != dt_addv2) {
    VLOG(2) << "MusaPReluFusion: dtype mismatch";
    return result;
  }

  // ===========================================================================
  // Match successful!
  // ===========================================================================

  result.matched = true;
  // Note: Select node is NOT included in matched_nodes - fusion starts from
  // after Select node, Select will be preserved
  result.matched_nodes.push_back(relu1_node);
  result.matched_nodes.push_back(neg2_node);
  result.matched_nodes.push_back(relu2_node);
  result.matched_nodes.push_back(const_node);
  result.matched_nodes.push_back(mul_node);
  result.matched_nodes.push_back(&addv2_node);
  // x_input: Select node's output (Select is preserved, not fused)
  // The fused MusaPRelu will take Select's output as its x input
  if (select_node) {
    result.captured_nodes["input"] = select_node;
  }

  result.captured_nodes["alpha_input"] = const_node;  // alpha const input

  result.captured_nodes["relu1"] = relu1_node;
  result.captured_nodes["neg2"] = neg2_node;
  result.captured_nodes["relu2"] = relu2_node;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["output"] = &addv2_node;

  VLOG(1) << "MusaPRelu: Matched PRelu pattern at AddV2 node '" << name << "'"
          << " (x from Select: " << select_node->name()
          << ", alpha: " << const_node->name() << ")";

  return result;
}

Status MusaPReluFusion::Apply(GraphDef* graph,
                              const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid PRelu match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  // ---- Retrieve captured nodes ----
  auto output_it = match_result.captured_nodes.find("output");
  auto input_it = match_result.captured_nodes.find("input");
  auto alpha_input_it = match_result.captured_nodes.find("alpha_input");

  if (output_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing output node in PRelu pattern");
  }

  const NodeDef* output_node = output_it->second;  // AddV2 node

  // ---- Duplicate-fusion guard ----
  std::string base_name = output_node->name();
  if (base_name.size() > 9 &&
      base_name.substr(base_name.size() - 9) == "_original") {
    base_name = base_name.substr(0, base_name.size() - 9);
  }

  for (const auto& node : graph->node()) {
    if (node.name() == base_name && node.op() == "MusaPRelu") {
      VLOG(1) << "MusaPRelu: " << base_name << " is already fused, skipping";
      return Status::OK();
    }
  }

  // ---- Create fused MusaPRelu node ----
  std::string fused_node_name = output_node->name() + "_fused_prelu";
  VLOG(1) << "MusaPRelu: Creating fused node: " << fused_node_name;

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaPRelu");
  fused_node->set_device(output_node->device());

  // Input: Select node output (Select is preserved, not fused)
  // The fused MusaPRelu takes Select's output as its x input
  if (input_it != match_result.captured_nodes.end() && input_it->second) {
    // Use Select node name as input (preserves the connection)
    fused_node->add_input(input_it->second->name());
  } else {
    return Status(error::INVALID_ARGUMENT, "Cannot determine PRelu x input");
  }

  // Alpha input: Const node
  if (alpha_input_it != match_result.captured_nodes.end() &&
      alpha_input_it->second) {
    fused_node->add_input(alpha_input_it->second->name());
  } else {
    return Status(error::INVALID_ARGUMENT,
                  "Cannot determine PRelu alpha input");
  }

  // Attributes
  auto* attr = fused_node->mutable_attr();

  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    (*attr)["T"] = dtype_it->second;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }

  // 1. Rename original output (AddV2) node
  std::string original_name = output_node->name();
  const_cast<NodeDef*>(output_node)->set_name(original_name + "_original");

  // 2. Rename fused node to original name (so downstream consumers reconnect)
  fused_node->set_name(original_name);
  VLOG(1) << "MusaPRelu: Fused node created as " << original_name;

  // 3. Collect nodes to remove
  // Note: Select node is NOT removed - fusion starts after Select
  std::set<std::string> nodes_to_remove;

  // The renamed AddV2 node (output_node)
  nodes_to_remove.insert(original_name + "_original");

  // Intermediate nodes in the pattern (excluding Select)
  auto relu1_it = match_result.captured_nodes.find("relu1");
  if (relu1_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(relu1_it->second->name());
  }

  auto neg2_it = match_result.captured_nodes.find("neg2");
  if (neg2_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(neg2_it->second->name());
  }

  auto relu2_it = match_result.captured_nodes.find("relu2");
  if (relu2_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(relu2_it->second->name());
  }

  auto mul_it = match_result.captured_nodes.find("mul");
  if (mul_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(mul_it->second->name());
  }

  // Const node (alpha) - only remove if no other consumers
  if (alpha_input_it != match_result.captured_nodes.end() &&
      alpha_input_it->second) {
    int const_consumers =
        CountConsumers(*graph, alpha_input_it->second->name());
    if (const_consumers <= 1) {
      nodes_to_remove.insert(alpha_input_it->second->name());
    }
  }

  // 4. Remove nodes from graph (from back to front to avoid index issues)
  std::vector<int> indices_to_remove;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_remove.count(graph->node(i).name()) > 0) {
      indices_to_remove.push_back(i);
    }
  }
  std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());

  for (int idx : indices_to_remove) {
    VLOG(2) << "MusaPRelu: Removing node: " << graph->node(idx).name();
    FusionGraphUtils::RemoveNode(graph, idx);
  }

  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaPReluFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaPReluFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
