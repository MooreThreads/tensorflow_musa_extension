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

#include "kernels/fusion/layernorm_fusion.h"

#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "mu/optimizer/graph_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
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

// Helper: check that an input is a Const (optionally through Identity or
// ExpandDims chain).
const NodeDef* GetConstLikeNode(const GraphDef& graph,
                                const std::string& input_name) {
  const NodeDef* node = FindProducer(graph, input_name);
  if (!node) return nullptr;

  // Allow Identity chain: Const -> Identity -> Identity ...
  while (node && IsOp(*node, "Identity")) {
    if (node->input_size() == 0) return nullptr;
    node = FindProducer(graph, node->input(0));
  }

  // Allow ExpandDims: Const -> ExpandDims (for gamma/beta expansion)
  if (node && IsOp(*node, "ExpandDims")) {
    if (node->input_size() < 1) return nullptr;
    node = FindProducer(graph, node->input(0));
    // Skip any Identity after ExpandDims
    while (node && IsOp(*node, "Identity")) {
      if (node->input_size() == 0) return nullptr;
      node = FindProducer(graph, node->input(0));
    }
  }

  if (!node || !IsOp(*node, "Const")) return nullptr;
  return node;
}

// Helper: extract float scalar value from a Const node
bool ExtractFloatScalar(const NodeDef* const_node, float* out_value) {
  if (!const_node || !out_value) return false;

  auto value_it = const_node->attr().find("value");
  if (value_it == const_node->attr().end()) return false;

  const TensorProto& tp = value_it->second.tensor();
  if (tp.dtype() != DT_FLOAT) return false;

  if (tp.float_val_size() > 0) {
    *out_value = tp.float_val(0);
    return true;
  }
  if (!tp.tensor_content().empty()) {
    const float* data =
        reinterpret_cast<const float*>(tp.tensor_content().data());
    *out_value = data[0];
    return true;
  }
  return false;
}

// Find LayerNorm prefix from node name
// Example: "fwffm_pbp_mlp/ad_emb_aug_ln_layer/add_1" ->
// "fwffm_pbp_mlp/ad_emb_aug_ln_layer" Rule: Extract from the beginning to the
// last '/' (excluding the last segment)
std::string FindLayerNormPrefix(const std::string& node_name) {
  if (node_name.empty()) return "";

  // Find the last '/' in the node name
  size_t last_slash_pos = node_name.rfind('/');
  if (last_slash_pos == std::string::npos) {
    // No '/' found, return empty (no prefix)
    return "";
  }

  // Extract prefix: from beginning to the last '/'
  return node_name.substr(0, last_slash_pos);
}

// Check if node belongs to the same LayerNorm subgraph
bool BelongsToLayerNorm(const std::string& node_name,
                        const std::string& prefix) {
  if (prefix.empty()) return false;
  if (node_name == prefix) return true;
  return node_name.length() > prefix.length() &&
         node_name.compare(0, prefix.length(), prefix) == 0 &&
         node_name[prefix.length()] == '/';
}

}  // namespace

// =============================================================================
// MusaLayerNormFusion Implementation
// =============================================================================
//
// Pattern to match:
//   Layer 1 (start): MusaNormalize (output: normalized tensor)
//   Layer 2:         Mul        - Scale by gamma
//   Layer 3 (end):   AddV2      - Add beta bias
//
// After fusion: MusaLayerNorm(x, gamma, beta)
//

MusaLayerNormFusion::MusaLayerNormFusion() = default;

bool MusaLayerNormFusion::IsKernelAvailable() const {
  if (IsDisabledByEnv()) {
    return false;
  }
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaLayerNormFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    VLOG(2) << "[LayerNorm::Match] RETURN empty: node_idx out of range";
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  if (IsOp(start_node, "AddV2")) {
    VLOG(2) << "[LayerNorm::Match] ENTER AddV2 path, node="
            << start_node.name();
    return MatchFromAddNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaLayerNormFusion::MatchFromAddNode(
    const GraphDef& graph, int add_node_idx) const {
  FusionMatchResult result;
  const NodeDef& add_node = graph.node(add_node_idx);

  VLOG(2) << "[LayerNorm::Match] MatchFromAddNode ENTER, node="
          << add_node.name();

  if (!IsOp(add_node, "AddV2")) {
    VLOG(2) << "[LayerNorm::Match] FAIL: not AddV2 op, node="
            << add_node.name();
    return result;
  }

  // Extract LayerNorm prefix
  const std::string layernorm_prefix = FindLayerNormPrefix(add_node.name());
  if (layernorm_prefix.empty()) {
    VLOG(2) << "[LayerNorm::Match] FAIL: cannot extract LayerNorm prefix, node="
            << add_node.name();
    return result;
  }
  VLOG(2) << "[LayerNorm::Match] prefix=" << layernorm_prefix;

  // =========================================================================
  // Layer 3 (end): AddV2 inputs:
  //   - input[0]: Mul (layer 2) OR beta (Const/ExpandDims)
  //   - input[1]: beta (Const/ExpandDims) OR Mul (layer 2)
  // =========================================================================
  if (add_node.input_size() != 2) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer3: AddV2 input_size="
            << add_node.input_size() << " (need 2), node=" << add_node.name();
    return result;
  }

  const NodeDef* add_input0 = FindProducer(graph, add_node.input(0));
  const NodeDef* add_input1 = FindProducer(graph, add_node.input(1));

  if (!add_input0 || !add_input1) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer3: cannot find AddV2 inputs, node="
            << add_node.name();
    return result;
  }

  // Determine which input is Mul (layer 2) and which is beta
  const NodeDef* mul_node = nullptr;
  const NodeDef* beta_input = nullptr;
  std::string beta_input_name;

  if (IsOp(*add_input0, "Mul") &&
      BelongsToLayerNorm(add_input0->name(), layernorm_prefix)) {
    mul_node = add_input0;
    beta_input = add_input1;
    beta_input_name = GetCleanName(add_node.input(1));
  } else if (IsOp(*add_input1, "Mul") &&
             BelongsToLayerNorm(add_input1->name(), layernorm_prefix)) {
    mul_node = add_input1;
    beta_input = add_input0;
    beta_input_name = GetCleanName(add_node.input(0));
  } else {
    VLOG(2)
        << "[LayerNorm::Match] FAIL layer3: AddV2 inputs are not (Mul, beta), "
        << "input0=" << add_input0->op() << ", input1=" << add_input1->op()
        << ", node=" << add_node.name();
    return result;
  }

  // Verify beta is a Const or ExpandDims of Const
  const NodeDef* beta_const = GetConstLikeNode(graph, beta_input_name);
  if (!beta_const) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer3: beta is not Const, node="
            << add_node.name();
    return result;
  }

  VLOG(2) << "[LayerNorm::Match] PASS layer3: Mul=" << mul_node->name()
          << ", beta=" << beta_const->name();

  // =========================================================================
  // Layer 2: Mul inputs:
  //   - input[0]: MusaNormalize (layer 1) OR gamma (Const/ExpandDims)
  //   - input[1]: gamma (Const/ExpandDims) OR MusaNormalize (layer 1)
  // =========================================================================
  if (mul_node->input_size() != 2) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer2: Mul input_size="
            << mul_node->input_size() << ", node=" << add_node.name();
    return result;
  }

  const NodeDef* mul_input0 = FindProducer(graph, mul_node->input(0));
  const NodeDef* mul_input1 = FindProducer(graph, mul_node->input(1));

  if (!mul_input0 || !mul_input1) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer2: cannot find Mul inputs, node="
            << add_node.name();
    return result;
  }

  // Determine which input is MusaNormalize (layer 1) and which is gamma
  const NodeDef* normalize_node = nullptr;
  const NodeDef* gamma_input = nullptr;
  std::string gamma_input_name;

  if (IsOp(*mul_input0, "MusaNormalize") &&
      BelongsToLayerNorm(mul_input0->name(), layernorm_prefix)) {
    normalize_node = mul_input0;
    gamma_input = mul_input1;
    gamma_input_name = GetCleanName(mul_node->input(1));
  } else if (IsOp(*mul_input1, "MusaNormalize") &&
             BelongsToLayerNorm(mul_input1->name(), layernorm_prefix)) {
    normalize_node = mul_input1;
    gamma_input = mul_input0;
    gamma_input_name = GetCleanName(mul_node->input(0));
  } else {
    VLOG(2) << "[LayerNorm::Match] FAIL layer2: Mul inputs are not "
               "(MusaNormalize, gamma), "
            << "input0=" << mul_input0->op() << ", input1=" << mul_input1->op()
            << ", node=" << add_node.name();
    return result;
  }

  // Verify gamma is a Const or ExpandDims of Const
  const NodeDef* gamma_const = GetConstLikeNode(graph, gamma_input_name);
  if (!gamma_const) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer2: gamma is not Const, node="
            << add_node.name();
    return result;
  }

  VLOG(2) << "[LayerNorm::Match] PASS layer2: MusaNormalize="
          << normalize_node->name() << ", gamma=" << gamma_const->name();

  // =========================================================================
  // Layer 1 (start): MusaNormalize inputs:
  //   - input[0]: x (original input)
  //   - input[1]: gamma_default (Const, typically 1.0)
  //   - input[2]: beta_default (Const, typically 0.0)
  // =========================================================================
  if (normalize_node->input_size() < 1) {
    VLOG(2) << "[LayerNorm::Match] FAIL layer1: MusaNormalize input_size="
            << normalize_node->input_size() << ", node=" << add_node.name();
    return result;
  }

  const NodeDef* original_input = FindProducer(graph, normalize_node->input(0));
  if (!original_input) {
    VLOG(2)
        << "[LayerNorm::Match] FAIL layer1: cannot find original input, node="
        << add_node.name();
    return result;
  }

  // Get epsilon from MusaNormalize attribute
  float epsilon = 1e-5f;
  auto epsilon_attr = normalize_node->attr().find("epsilon");
  if (epsilon_attr != normalize_node->attr().end()) {
    epsilon = epsilon_attr->second.f();
  }

  VLOG(2) << "[LayerNorm::Match] PASS layer1: original_input="
          << original_input->name() << ", epsilon=" << epsilon;

  // =========================================================================
  // Check for external references (fork detection)
  // If any matched node (except output) is referenced by external nodes,
  // we cannot safely fuse this subgraph.
  // =========================================================================
  std::unordered_set<std::string> matched_node_names;
  matched_node_names.insert(add_node.name());
  matched_node_names.insert(mul_node->name());
  matched_node_names.insert(normalize_node->name());

  // The output node (add_node) can be referenced externally - that's the normal
  // case We only check if intermediate nodes are referenced externally
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& node = graph.node(i);
    // Skip nodes that are part of the matched subgraph
    if (matched_node_names.count(node.name())) continue;

    // Check all inputs of this external node
    for (int j = 0; j < node.input_size(); ++j) {
      std::string input_name = GetCleanName(node.input(j));
      // If this external node references an intermediate matched node (not the
      // output)
      if (input_name != add_node.name() &&
          matched_node_names.count(input_name)) {
        VLOG(1) << "[LayerNorm::Match] REJECT: fork detected, node "
                << input_name << " is referenced by external node "
                << node.name();
        return result;  // Return empty result - cannot fuse
      }
    }
  }

  // =========================================================================
  // Build match result
  // =========================================================================
  result.matched = true;

  // Add all matched nodes (from end to start)
  result.matched_nodes.push_back(&add_node);       // Layer 3
  result.matched_nodes.push_back(mul_node);        // Layer 2
  result.matched_nodes.push_back(normalize_node);  // Layer 1

  // Store captured nodes
  result.captured_nodes["output"] = &add_node;
  result.captured_nodes["add"] = &add_node;
  result.captured_nodes["mul"] = mul_node;
  result.captured_nodes["normalize"] = normalize_node;

  // Store captured attributes
  result.captured_attrs["original_input"] = original_input->name();
  result.captured_attrs["layernorm_prefix"] = layernorm_prefix;

  // Store gamma and beta names (may be Const or ExpandDims)
  result.captured_attrs["gamma_input"] = gamma_input_name;
  result.captured_attrs["gamma_const"] = gamma_const->name();
  result.captured_attrs["beta_input"] = beta_input_name;
  result.captured_attrs["beta_const"] = beta_const->name();

  // Store epsilon from MusaNormalize
  std::ostringstream epsilon_ss;
  epsilon_ss << epsilon;
  result.captured_attrs["epsilon"] = epsilon_ss.str();

  // Record all matched nodes for deletion
  for (const NodeDef* matched_node : result.matched_nodes) {
    result.captured_attrs["fuse_node_" +
                          std::to_string(result.captured_attrs.size())] =
        matched_node->name();
  }

  VLOG(1) << "[LayerNorm::Match] SUCCESS matched=" << add_node.name()
          << ", input=" << original_input->name()
          << ", gamma=" << gamma_const->name()
          << ", beta=" << beta_const->name() << ", epsilon=" << epsilon
          << ", prefix=" << layernorm_prefix
          << ", fuse_nodes=" << result.matched_nodes.size();

  return result;
}

Status MusaLayerNormFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  VLOG(2) << "[LayerNorm::Apply] ENTER, matched=" << match_result.matched
          << ", nodes_count=" << match_result.matched_nodes.size()
          << ", kernel_available=" << IsKernelAvailable();

  if (!match_result.IsValid()) {
    VLOG(2) << "[LayerNorm::Apply] RETURN: invalid match result";
    return Status(error::INVALID_ARGUMENT, "Invalid LayerNorm match result");
  }

  if (!IsKernelAvailable()) {
    VLOG(2)
        << "[LayerNorm::Apply] RETURN: kernel not available, skipping fusion";
    return Status::OK();
  }

  // Get the output node (AddV2)
  auto output_it = match_result.captured_nodes.find("output");
  if (output_it == match_result.captured_nodes.end()) {
    VLOG(2)
        << "[LayerNorm::Apply] RETURN: missing output node in captured_nodes";
    return Status(error::INVALID_ARGUMENT,
                  "Missing output node in LayerNorm pattern");
  }

  const NodeDef* output_node = output_it->second;
  std::string output_name = output_node->name();
  std::string output_device = output_node->device();
  VLOG(2) << "[LayerNorm::Apply] output_node=" << output_name;

  // Check if already fused
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaLayerNorm") {
      VLOG(2) << "[LayerNorm::Apply] RETURN: already fused, node="
              << output_name;
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  // Get input name
  std::string input_name;
  auto original_input_it = match_result.captured_attrs.find("original_input");
  if (original_input_it != match_result.captured_attrs.end() &&
      !original_input_it->second.empty()) {
    input_name = original_input_it->second;
  } else {
    VLOG(2) << "[LayerNorm::Apply] RETURN: cannot determine input";
    return Status(error::INVALID_ARGUMENT, "Cannot determine LayerNorm input");
  }

  // Get data type
  DataType dtype = DT_FLOAT;
  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    dtype = dtype_it->second.type();
  }

  // Extract epsilon
  float epsilon = 1e-5f;
  auto epsilon_it = match_result.captured_attrs.find("epsilon");
  if (epsilon_it != match_result.captured_attrs.end()) {
    epsilon = std::stof(epsilon_it->second);
  }

  // Get gamma and beta input names
  std::string gamma_input_name;
  std::string beta_input_name;
  auto gamma_input_it = match_result.captured_attrs.find("gamma_input");
  auto beta_input_it = match_result.captured_attrs.find("beta_input");
  if (gamma_input_it != match_result.captured_attrs.end()) {
    gamma_input_name = gamma_input_it->second;
  }
  if (beta_input_it != match_result.captured_attrs.end()) {
    beta_input_name = beta_input_it->second;
  }

  VLOG(2) << "[LayerNorm::Apply] input=" << input_name
          << ", gamma=" << gamma_input_name << ", beta=" << beta_input_name
          << ", epsilon=" << epsilon;

  // =========================================================================
  // Collect nodes to delete (from captured_attrs fuse_node_*)
  // =========================================================================
  std::unordered_set<std::string> fuse_node_names;
  for (const auto& kv : match_result.captured_attrs) {
    if (kv.first.substr(0, 10) == "fuse_node_") {
      fuse_node_names.insert(kv.second);
    }
  }

  // Don't delete the input nodes
  fuse_node_names.erase(input_name);
  fuse_node_names.erase(gamma_input_name);
  fuse_node_names.erase(beta_input_name);

  // =========================================================================
  // Check for shared nodes (referenced by external nodes)
  // =========================================================================
  std::unordered_set<std::string> shared_nodes;

  VLOG(1) << "[LayerNorm::Apply] Checking for shared nodes, graph has "
          << graph->node_size() << " nodes, fuse_node_names has "
          << fuse_node_names.size() << " nodes to delete";

  for (int i = 0; i < graph->node_size(); ++i) {
    const NodeDef& node = graph->node(i);
    // Skip nodes in the deletion list
    if (fuse_node_names.count(node.name())) continue;

    // Check all inputs of this node
    for (int j = 0; j < node.input_size(); ++j) {
      std::string producer =
          FusionGraphUtils::GetProducerNodeName(node.input(j));
      // If input comes from a node in the deletion list (and not the output
      // node)
      if (fuse_node_names.count(producer) && producer != output_name) {
        shared_nodes.insert(producer);
        VLOG(1) << "[LayerNorm::Apply] SHARED NODE DETECTED: " << producer
                << " (referenced by external node: " << node.name() << ")";
      }
    }
  }

  // Remove shared nodes from deletion list
  for (const auto& name : shared_nodes) {
    VLOG(1) << "[LayerNorm::Apply] KEEPING SHARED NODE: " << name;
    fuse_node_names.erase(name);
  }

  VLOG(2) << "[LayerNorm::Apply] will remove " << fuse_node_names.size()
          << " fused sub-graph nodes, found " << shared_nodes.size()
          << " shared nodes";

  // =========================================================================
  // Delete fused nodes
  // =========================================================================
  int removed_count = 0;
  for (auto it = fuse_node_names.begin(); it != fuse_node_names.end();) {
    int idx = FusionGraphUtils::FindNodeIndex(*graph, *it);
    if (idx >= 0) {
      FusionGraphUtils::RemoveNode(graph, idx);
      removed_count++;
      it = fuse_node_names.erase(it);
    } else {
      ++it;
    }
  }

  VLOG(2) << "[LayerNorm::Apply] removed " << removed_count << " nodes";

  // =========================================================================
  // Create fused node (MusaLayerNorm)
  // Inputs: x (original input), gamma, beta
  // =========================================================================
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(output_name);
  fused_node->set_op("MusaLayerNorm");
  fused_node->set_device(output_device);

  // Add inputs: input, gamma, beta
  fused_node->add_input(input_name);
  fused_node->add_input(gamma_input_name);
  fused_node->add_input(beta_input_name);

  // Set attributes
  auto* attr = fused_node->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["epsilon"].set_f(epsilon);

  VLOG(1) << "[LayerNorm::Apply] SUCCESS fused to " << output_name
          << ", input=" << input_name << ", gamma=" << gamma_input_name
          << ", beta=" << beta_input_name << ", epsilon=" << epsilon
          << ", removed=" << removed_count
          << ", graph_nodes=" << graph->node_size();

  return Status::OK();
}

// Register fusion pattern
REGISTER_FUSION_PATTERN(MusaLayerNormFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaLayerNormFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
