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

#include "mu/graph_fusion/normalize_fusion.h"

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

// Helper to build a name->NodeDef* map for fast lookup
std::unordered_map<std::string, const NodeDef*> BuildNodeMap(
    const GraphDef& graph) {
  std::unordered_map<std::string, const NodeDef*> m;
  m.reserve(graph.node_size());
  for (int i = 0; i < graph.node_size(); ++i) {
    m.emplace(graph.node(i).name(), &graph.node(i));
  }
  return m;
}

// Helper: check that an input is a Const (optionally through Identity chain).
const NodeDef* GetConstLikeNode(const GraphDef& graph,
                                const std::string& input_name) {
  const NodeDef* node = FindProducer(graph, input_name);
  if (!node) return nullptr;

  // Allow Identity chain: Const -> Identity -> Identity ...
  while (node && IsOp(*node, "Identity")) {
    if (node->input_size() == 0) return nullptr;
    node = FindProducer(graph, node->input(0));
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

// Extract reduction indices from Mean node's second input (Const)
std::vector<int> ExtractReductionIndices(const GraphDef& graph,
                                         const NodeDef* mean_node) {
  std::vector<int> indices;
  if (!mean_node || mean_node->input_size() < 2) return indices;

  const NodeDef* indices_const = GetConstLikeNode(graph, mean_node->input(1));
  if (!indices_const) return indices;

  auto value_it = indices_const->attr().find("value");
  if (value_it == indices_const->attr().end()) return indices;

  const TensorProto& tp = value_it->second.tensor();

  if (tp.dtype() == DT_INT32) {
    if (tp.int_val_size() > 0) {
      for (int i = 0; i < tp.int_val_size(); ++i) {
        indices.push_back(tp.int_val(i));
      }
    } else if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int32_t);
      const int32_t* data =
          reinterpret_cast<const int32_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        indices.push_back(data[i]);
      }
    }
  } else if (tp.dtype() == DT_INT64) {
    if (tp.int64_val_size() > 0) {
      for (int i = 0; i < tp.int64_val_size(); ++i) {
        indices.push_back(static_cast<int>(tp.int64_val(i)));
      }
    } else if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int64_t);
      const int64_t* data =
          reinterpret_cast<const int64_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        indices.push_back(static_cast<int>(data[i]));
      }
    }
  }

  // Scalar tensor (rank-0)
  if (indices.empty() && tp.tensor_shape().dim_size() == 0) {
    if (tp.dtype() == DT_INT32 && tp.int_val_size() == 1) {
      indices.push_back(tp.int_val(0));
    } else if (tp.dtype() == DT_INT64 && tp.int64_val_size() == 1) {
      indices.push_back(static_cast<int>(tp.int64_val(0)));
    }
  }

  return indices;
}

// Serialize reduction indices to comma-separated string
std::string IndicesToString(const std::vector<int>& indices) {
  std::ostringstream oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) oss << ",";
    oss << indices[i];
  }
  return oss.str();
}

// Parse comma-separated indices string back to vector
std::vector<int> StringToIndices(const std::string& s) {
  std::vector<int> indices;
  if (s.empty()) return indices;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    indices.push_back(std::stoi(token));
  }
  return indices;
}

// Find Normalize prefix from node name
// Example: "fwffm_pbp_mlp/ad_emb_aug_ln_layer/truediv" ->
// "fwffm_pbp_mlp/ad_emb_aug_ln_layer" Rule: Extract from the beginning to the
// last '/' (excluding the last segment)
std::string FindNormalizePrefix(const std::string& node_name) {
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

// Check if node belongs to the same Normalize subgraph
bool BelongsToNormalize(const std::string& node_name,
                        const std::string& prefix) {
  if (prefix.empty()) return false;
  if (node_name == prefix) return true;
  return node_name.length() > prefix.length() &&
         node_name.compare(0, prefix.length(), prefix) == 0 &&
         node_name[prefix.length()] == '/';
}

// Collect all nodes with a given prefix
std::vector<std::string> CollectNodesWithPrefix(const GraphDef& graph,
                                                const std::string& prefix) {
  std::vector<std::string> nodes;
  if (prefix.empty()) return nodes;

  for (int i = 0; i < graph.node_size(); ++i) {
    const std::string& name = graph.node(i).name();
    if (name == prefix || (name.length() > prefix.length() &&
                           name.substr(0, prefix.length()) == prefix &&
                           name[prefix.length()] == '/')) {
      nodes.push_back(name);
    }
  }
  return nodes;
}

}  // namespace

// =============================================================================
// MusaNormalizeFusion Implementation
// =============================================================================

MusaNormalizeFusion::MusaNormalizeFusion() = default;

bool MusaNormalizeFusion::IsKernelAvailable() const {
  if (IsDisabledByEnv()) {
    return false;
  }
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaNormalizeFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    VLOG(2) << "[Normalize::Match] RETURN empty: node_idx out of range";
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  if (IsOp(start_node, "RealDiv")) {
    VLOG(2) << "[Normalize::Match] ENTER RealDiv path, node="
            << start_node.name();
    return MatchFromRealDivNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaNormalizeFusion::MatchFromRealDivNode(
    const GraphDef& graph, int realdiv_node_idx) const {
  FusionMatchResult result;
  const NodeDef& realdiv_node = graph.node(realdiv_node_idx);

  VLOG(2) << "[Normalize::Match] MatchFromRealDivNode ENTER, node="
          << realdiv_node.name();

  if (!IsOp(realdiv_node, "RealDiv")) {
    VLOG(2) << "[Normalize::Match] FAIL: not RealDiv op, node="
            << realdiv_node.name();
    return result;
  }

  // Extract Normalize prefix
  const std::string normalize_prefix = FindNormalizePrefix(realdiv_node.name());
  if (normalize_prefix.empty()) {
    VLOG(2) << "[Normalize::Match] FAIL: cannot extract Normalize prefix, node="
            << realdiv_node.name();
    return result;
  }
  VLOG(2) << "[Normalize::Match] prefix=" << normalize_prefix;

  // Build node map for fast lookup
  auto node_map = BuildNodeMap(graph);

  // =========================================================================
  // 第1步: RealDiv 的两个输入:
  //   - input[0]: Sub (属于前缀)
  //   - input[1]: MusaClip (属于前缀)
  // =========================================================================
  if (realdiv_node.input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step1: RealDiv input_size="
            << realdiv_node.input_size()
            << " (need 2), node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* sub_node = FindProducer(graph, realdiv_node.input(0));
  const NodeDef* clip_node = FindProducer(graph, realdiv_node.input(1));

  if (!sub_node || !IsOp(*sub_node, "Sub")) {
    VLOG(2) << "[Normalize::Match] FAIL step1: input[0] is not Sub, actual="
            << (sub_node ? sub_node->op() : "NULL")
            << ", node=" << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(sub_node->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step1: Sub not in prefix, name="
            << sub_node->name() << ", prefix=" << normalize_prefix;
    return result;
  }
  if (!clip_node || !IsOp(*clip_node, "MusaClip")) {
    VLOG(2)
        << "[Normalize::Match] FAIL step1: input[1] is not MusaClip, actual="
        << (clip_node ? clip_node->op() : "NULL")
        << ", node=" << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(clip_node->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step1: MusaClip not in prefix, name="
            << clip_node->name() << ", prefix=" << normalize_prefix;
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step1: Sub=" << sub_node->name()
          << ", MusaClip=" << clip_node->name();

  // =========================================================================
  // 第2步: MusaClip 的输入:
  //   - input[0]: Sqrt (属于前缀)
  //   - input[1]: clip_min (Const, 共享)
  //   - input[2]: clip_max (Const, 共享)
  // =========================================================================
  if (clip_node->input_size() < 3) {
    VLOG(2) << "[Normalize::Match] FAIL step2: MusaClip input_size="
            << clip_node->input_size()
            << " (need >=3), node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* sqrt_node = FindProducer(graph, clip_node->input(0));
  if (!sqrt_node || !IsOp(*sqrt_node, "Sqrt")) {
    VLOG(2)
        << "[Normalize::Match] FAIL step2: MusaClip input[0] is not Sqrt, node="
        << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(sqrt_node->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step2: Sqrt not in prefix, name="
            << sqrt_node->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step2: Sqrt=" << sqrt_node->name();

  // =========================================================================
  // 第3步: Sqrt 的输入: ExpandDims_2 (属于前缀)
  // =========================================================================
  if (sqrt_node->input_size() != 1) {
    VLOG(2) << "[Normalize::Match] FAIL step3: Sqrt input_size="
            << sqrt_node->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* expanddims_2 = FindProducer(graph, sqrt_node->input(0));
  if (!expanddims_2 || !IsOp(*expanddims_2, "ExpandDims")) {
    VLOG(2)
        << "[Normalize::Match] FAIL step3: Sqrt input is not ExpandDims, node="
        << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(expanddims_2->name(), normalize_prefix)) {
    VLOG(2)
        << "[Normalize::Match] FAIL step3: ExpandDims_2 not in prefix, name="
        << expanddims_2->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step3: ExpandDims_2="
          << expanddims_2->name();

  // =========================================================================
  // 第4步: ExpandDims_2 的输入:
  //   - input[0]: Mean_2 (属于前缀)
  //   - input[1]: dim (Const, 共享)
  // =========================================================================
  if (expanddims_2->input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step4: ExpandDims_2 input_size="
            << expanddims_2->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* mean_2 = FindProducer(graph, expanddims_2->input(0));
  if (!mean_2 || !IsOp(*mean_2, "Mean")) {
    VLOG(2) << "[Normalize::Match] FAIL step4: ExpandDims_2 input[0] is not "
               "Mean, node="
            << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(mean_2->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step4: Mean_2 not in prefix, name="
            << mean_2->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step4: Mean_2=" << mean_2->name();

  // =========================================================================
  // 第5步: Mean_2 的输入:
  //   - input[0]: Square (属于前缀)
  //   - input[1]: reduction_indices (Const, 共享)
  // =========================================================================
  if (mean_2->input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step5: Mean_2 input_size="
            << mean_2->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* square_node = FindProducer(graph, mean_2->input(0));
  if (!square_node || !IsOp(*square_node, "Square")) {
    VLOG(2)
        << "[Normalize::Match] FAIL step5: Mean_2 input[0] is not Square, node="
        << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(square_node->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step5: Square not in prefix, name="
            << square_node->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step5: Square=" << square_node->name();

  // =========================================================================
  // 第6步: Square 的输入: Sub (已找到，与 RealDiv input[0] 相同)
  // =========================================================================
  if (square_node->input_size() != 1) {
    VLOG(2) << "[Normalize::Match] FAIL step6: Square input_size="
            << square_node->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* square_input = FindProducer(graph, square_node->input(0));
  if (square_input != sub_node) {
    VLOG(2) << "[Normalize::Match] FAIL step6: Square input is not the same "
               "Sub, node="
            << realdiv_node.name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step6: Square input matches Sub";

  // =========================================================================
  // 第7步: Sub 的输入:
  //   - input[0]: 原始输入 (外部节点，如 BiasAdd)
  //   - input[1]: ExpandDims_1 (属于前缀)
  // =========================================================================
  if (sub_node->input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step7: Sub input_size="
            << sub_node->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* original_input = FindProducer(graph, sub_node->input(0));
  const NodeDef* expanddims_1 = FindProducer(graph, sub_node->input(1));

  if (!original_input) {
    VLOG(2)
        << "[Normalize::Match] FAIL step7: cannot find original input, node="
        << realdiv_node.name();
    return result;
  }
  if (!expanddims_1 || !IsOp(*expanddims_1, "ExpandDims")) {
    VLOG(2) << "[Normalize::Match] FAIL step7: Sub input[1] is not ExpandDims, "
               "node="
            << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(expanddims_1->name(), normalize_prefix)) {
    VLOG(2)
        << "[Normalize::Match] FAIL step7: ExpandDims_1 not in prefix, name="
        << expanddims_1->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step7: original_input="
          << original_input->name()
          << ", ExpandDims_1=" << expanddims_1->name();

  // =========================================================================
  // 第8步: ExpandDims_1 的输入:
  //   - input[0]: Mean_1 (属于前缀)
  //   - input[1]: dim (Const, 共享)
  // =========================================================================
  if (expanddims_1->input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step8: ExpandDims_1 input_size="
            << expanddims_1->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* mean_1 = FindProducer(graph, expanddims_1->input(0));
  if (!mean_1 || !IsOp(*mean_1, "Mean")) {
    VLOG(2) << "[Normalize::Match] FAIL step8: ExpandDims_1 input[0] is not "
               "Mean, node="
            << realdiv_node.name();
    return result;
  }
  if (!BelongsToNormalize(mean_1->name(), normalize_prefix)) {
    VLOG(2) << "[Normalize::Match] FAIL step8: Mean_1 not in prefix, name="
            << mean_1->name();
    return result;
  }

  VLOG(2) << "[Normalize::Match] PASS step8: Mean_1=" << mean_1->name();

  // =========================================================================
  // 第9步: Mean_1 的输入:
  //   - input[0]: 原始输入 (外部节点，与 Sub input[0] 相同)
  //   - input[1]: reduction_indices (Const, 共享)
  // =========================================================================
  if (mean_1->input_size() != 2) {
    VLOG(2) << "[Normalize::Match] FAIL step9: Mean_1 input_size="
            << mean_1->input_size() << ", node=" << realdiv_node.name();
    return result;
  }

  const NodeDef* mean_1_input = FindProducer(graph, mean_1->input(0));
  if (mean_1_input != original_input) {
    VLOG(2) << "[Normalize::Match] FAIL step9: Mean_1 input[0] is not the same "
               "as original_input, node="
            << realdiv_node.name();
    return result;
  }

  VLOG(2)
      << "[Normalize::Match] PASS step9: Mean_1 input matches original_input";

  // =========================================================================
  // 提取 reduction_indices 和 epsilon
  // =========================================================================
  std::vector<int> reduction_indices = ExtractReductionIndices(graph, mean_1);
  if (reduction_indices.empty()) {
    VLOG(2) << "[Normalize::Match] WARNING: could not extract "
               "reduction_indices, using default";
    reduction_indices.push_back(-1);
  }

  // Extract epsilon (clip_min) and max_std (clip_max) from MusaClip
  // MusaClip inputs: [input, clip_min, clip_max]
  float epsilon = 1e-6f;
  float max_std = std::numeric_limits<float>::max();
  const NodeDef* clip_min_node = GetConstLikeNode(graph, clip_node->input(1));
  const NodeDef* clip_max_node = GetConstLikeNode(graph, clip_node->input(2));

  ExtractFloatScalar(clip_min_node, &epsilon);
  ExtractFloatScalar(clip_max_node, &max_std);

  // Ensure epsilon is positive for numerical stability
  if (epsilon <= 0.0f) epsilon = 1e-6f;

  VLOG(2) << "[Normalize::Match] reduction_indices="
          << IndicesToString(reduction_indices) << ", epsilon=" << epsilon
          << ", max_std=" << max_std;

  // =========================================================================
  // 构建匹配结果
  // 注意：只包含匹配模式中的节点，不包含所有前缀匹配的节点
  // =========================================================================
  result.matched = true;

  result.matched_nodes.push_back(&realdiv_node);
  result.matched_nodes.push_back(sub_node);
  result.matched_nodes.push_back(clip_node);
  result.matched_nodes.push_back(sqrt_node);
  result.matched_nodes.push_back(expanddims_2);
  result.matched_nodes.push_back(mean_2);
  result.matched_nodes.push_back(square_node);
  result.matched_nodes.push_back(expanddims_1);
  result.matched_nodes.push_back(mean_1);

  result.captured_nodes["output"] = &realdiv_node;
  result.captured_nodes["sub"] = sub_node;
  result.captured_nodes["clip"] = clip_node;
  result.captured_nodes["sqrt"] = sqrt_node;
  result.captured_nodes["expanddims_2"] = expanddims_2;
  result.captured_nodes["mean_2"] = mean_2;
  result.captured_nodes["square"] = square_node;
  result.captured_nodes["expanddims_1"] = expanddims_1;
  result.captured_nodes["mean_1"] = mean_1;

  result.captured_attrs["original_input"] = original_input->name();
  result.captured_attrs["normalize_prefix"] = normalize_prefix;
  result.captured_attrs["reduction_indices"] =
      IndicesToString(reduction_indices);

  // Store epsilon as a string with scientific notation to preserve small values
  std::ostringstream epsilon_ss;
  epsilon_ss << epsilon;
  result.captured_attrs["epsilon"] = epsilon_ss.str();

  // Store max_std as a string
  std::ostringstream max_std_ss;
  max_std_ss << max_std;
  result.captured_attrs["max_std"] = max_std_ss.str();

  // 只记录匹配到的节点，不收集所有前缀匹配的节点
  // 这样可以避免删除不属于模式的其他节点（如 ExpandDims_3, add_1 等）
  for (const NodeDef* matched_node : result.matched_nodes) {
    result.captured_attrs["fuse_node_" +
                          std::to_string(result.captured_attrs.size())] =
        matched_node->name();
  }

  VLOG(1) << "[Normalize::Match] SUCCESS matched=" << realdiv_node.name()
          << ", input=" << original_input->name()
          << ", reduction_indices=" << IndicesToString(reduction_indices)
          << ", epsilon=" << epsilon << ", prefix=" << normalize_prefix
          << ", fuse_nodes=" << result.matched_nodes.size();

  return result;
}

Status MusaNormalizeFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  VLOG(2) << "[Normalize::Apply] ENTER, matched=" << match_result.matched
          << ", nodes_count=" << match_result.matched_nodes.size()
          << ", kernel_available=" << IsKernelAvailable();

  if (!match_result.IsValid()) {
    VLOG(2) << "[Normalize::Apply] RETURN: invalid match result";
    return Status(error::INVALID_ARGUMENT, "Invalid Normalize match result");
  }

  if (!IsKernelAvailable()) {
    VLOG(2)
        << "[Normalize::Apply] RETURN: kernel not available, skipping fusion";
    return Status::OK();
  }

  // 获取关键节点
  auto output_it = match_result.captured_nodes.find("output");

  if (output_it == match_result.captured_nodes.end()) {
    VLOG(2)
        << "[Normalize::Apply] RETURN: missing output node in captured_nodes";
    return Status(error::INVALID_ARGUMENT,
                  "Missing output node in Normalize pattern");
  }

  const NodeDef* output_node = output_it->second;
  std::string output_name = output_node->name();
  std::string output_device = output_node->device();
  VLOG(2) << "[Normalize::Apply] output_node=" << output_name;

  // 检查是否已经融合过
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaNormalize") {
      VLOG(2) << "[Normalize::Apply] RETURN: already fused, node="
              << output_name;
      return Status(error::ALREADY_EXISTS, "Already fused");
    }
  }

  // 获取输入名称
  std::string input_name;

  auto original_input_it = match_result.captured_attrs.find("original_input");
  if (original_input_it != match_result.captured_attrs.end() &&
      !original_input_it->second.empty()) {
    input_name = original_input_it->second;
  } else {
    VLOG(2) << "[Normalize::Apply] RETURN: cannot determine input";
    return Status(error::INVALID_ARGUMENT, "Cannot determine Normalize input");
  }

  // 获取数据类型
  DataType dtype = DT_FLOAT;
  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    dtype = dtype_it->second.type();
  }

  // 提取 reduction_indices 和 epsilon
  std::vector<int> reduction_indices;
  auto indices_it = match_result.captured_attrs.find("reduction_indices");
  if (indices_it != match_result.captured_attrs.end()) {
    reduction_indices = StringToIndices(indices_it->second);
  }
  if (reduction_indices.empty()) reduction_indices.push_back(-1);

  float epsilon = 1e-6f;
  auto epsilon_it = match_result.captured_attrs.find("epsilon");
  if (epsilon_it != match_result.captured_attrs.end()) {
    epsilon = std::stof(epsilon_it->second);
  }

  float max_std = std::numeric_limits<float>::max();
  auto max_std_it = match_result.captured_attrs.find("max_std");
  if (max_std_it != match_result.captured_attrs.end()) {
    max_std = std::stof(max_std_it->second);
  }

  VLOG(2) << "[Normalize::Apply] input=" << input_name
          << ", reduction_indices=" << IndicesToString(reduction_indices)
          << ", epsilon=" << epsilon;

  // =========================================================================
  // 收集需要删除的节点名称（从 captured_attrs 中的 fuse_node_* 提取）
  // =========================================================================
  std::unordered_set<std::string> fuse_node_names;
  for (const auto& kv : match_result.captured_attrs) {
    if (kv.first.substr(0, 10) == "fuse_node_") {
      fuse_node_names.insert(kv.second);
    }
  }

  fuse_node_names.erase(input_name);

  // =========================================================================
  // 检测每个节点是否被子图外部节点引用
  // 如果被引用，说明该节点被多个子图共享，不应该删除
  // =========================================================================
  std::unordered_set<std::string> shared_nodes;

  VLOG(1) << "[Normalize::Apply] Checking for shared nodes, graph has "
          << graph->node_size() << " nodes, fuse_node_names has "
          << fuse_node_names.size() << " nodes to delete";

  for (int i = 0; i < graph->node_size(); ++i) {
    const NodeDef& node = graph->node(i);
    // 跳过待删除节点本身
    if (fuse_node_names.count(node.name())) continue;

    // 检查该节点的所有输入
    for (int j = 0; j < node.input_size(); ++j) {
      std::string producer =
          FusionGraphUtils::GetProducerNodeName(node.input(j));
      // 如果输入来自待删除列表，说明该节点被子图外部引用
      if (fuse_node_names.count(producer) && producer != output_name) {
        shared_nodes.insert(producer);
        VLOG(1) << "[Normalize::Apply] SHARED NODE DETECTED: " << producer
                << " (referenced by external node: " << node.name() << ")";
      }
    }
  }

  // 从删除列表中移除共享节点
  for (const auto& name : shared_nodes) {
    VLOG(1) << "[Normalize::Apply] KEEPING SHARED NODE: " << name;
    fuse_node_names.erase(name);
  }

  // 记录共享节点，用于后续孤立节点检查时排除
  std::unordered_set<std::string> shared_const_nodes = shared_nodes;

  VLOG(2) << "[Normalize::Apply] will remove " << fuse_node_names.size()
          << " fused sub-graph nodes, found " << shared_const_nodes.size()
          << " shared nodes";

  // =========================================================================
  // 收集所有融合节点的输入节点（用于后续孤立节点检测）
  // =========================================================================
  std::unordered_set<std::string> potential_orphan_producers;
  for (const auto& node_name : fuse_node_names) {
    int idx = FusionGraphUtils::FindNodeIndex(*graph, node_name);
    if (idx >= 0 && idx < graph->node_size()) {
      const NodeDef& node = graph->node(idx);
      for (int j = 0; j < node.input_size(); ++j) {
        std::string producer =
            FusionGraphUtils::GetProducerNodeName(node.input(j));
        // 只记录不在融合子图内的输入节点
        if (producer != output_name && !fuse_node_names.count(producer) &&
            producer != input_name) {
          potential_orphan_producers.insert(producer);
        }
      }
    }
  }

  VLOG(2) << "[Normalize::Apply] found " << potential_orphan_producers.size()
          << " potential orphan producer nodes";

  // =========================================================================
  // 删除被融合的子图节点（融合节点继承 output_name，无需重定向）
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

  VLOG(2) << "[Normalize::Apply] removed " << removed_count << " nodes";

  // =========================================================================
  // 检查并删除孤立节点
  // 注意：不删除共享的 Const 节点（已在 shared_const_nodes 中记录）
  // 同时不删除任何 Const 节点，因为它们可能被其他子图使用
  // =========================================================================
  // int orphan_removed_count = 0;
  // for (auto it = potential_orphan_producers.begin();
  //      it != potential_orphan_producers.end();) {
  //   const std::string& producer_name = *it;

  //   // 跳过已知共享的 Const 节点
  //   if (shared_const_nodes.count(producer_name)) {
  //     VLOG(2) << "[Normalize::Apply] skipping shared const node: " <<
  //     producer_name;
  //     ++it;
  //     continue;
  //   }

  //   // 检查该节点是否还存在（可能已经被之前的删除操作删除）
  //   int idx = FusionGraphUtils::FindNodeIndex(*graph, producer_name);
  //   if (idx < 0 || idx >= graph->node_size()) {
  //     ++it;
  //     continue;
  //   }

  //   // 跳过 Const 节点 - 它们可能被图的其他部分使用
  //   const NodeDef& producer_node = graph->node(idx);
  //   if (IsOp(producer_node, "Const")) {
  //     VLOG(2) << "[Normalize::Apply] skipping Const node (may be shared): "
  //             << producer_name;
  //     ++it;
  //     continue;
  //   }

  //   // 检查该节点是否还有其他消费者
  //   bool has_consumers = false;
  //   for (int i = 0; i < graph->node_size(); ++i) {
  //     const NodeDef& node = graph->node(i);
  //     for (int j = 0; j < node.input_size(); ++j) {
  //       std::string input_producer =
  //           FusionGraphUtils::GetProducerNodeName(node.input(j));
  //       if (input_producer == producer_name) {
  //         has_consumers = true;
  //         break;
  //       }
  //     }
  //     if (has_consumers) break;
  //   }

  //   if (!has_consumers) {
  //     // 孤立节点，删除它
  //     VLOG(2) << "[Normalize::Apply] removing orphan node: " <<
  //     producer_name; FusionGraphUtils::RemoveNode(graph, idx);
  //     orphan_removed_count++;
  //     it = potential_orphan_producers.erase(it);
  //   } else {
  //     ++it;
  //   }
  // }

  // if (orphan_removed_count > 0) {
  //   VLOG(2) << "[Normalize::Apply] removed " << orphan_removed_count
  //           << " orphan nodes";
  // }

  // =========================================================================
  // 创建融合节点 (MusaNormalize)
  // 输入: x (原始输入)
  // 注意: MusaNormalize 需要 gamma 和 beta 输入
  //       对于无参数的 Normalize，传入标量 1.0 和 0.0
  // =========================================================================

  // 创建 gamma (默认为标量 1.0)
  NodeDef* gamma_node = graph->add_node();
  gamma_node->set_name(output_name + "/gamma");
  gamma_node->set_op("Const");
  gamma_node->set_device(output_device);
  auto* gamma_attr = gamma_node->mutable_attr();
  (*gamma_attr)["dtype"].set_type(dtype);
  TensorProto* gamma_tensor = (*gamma_attr)["value"].mutable_tensor();
  gamma_tensor->set_dtype(dtype);
  // 设置标量值 1.0
  if (dtype == DT_FLOAT) {
    gamma_tensor->add_float_val(1.0f);
  } else if (dtype == DT_HALF) {
    Eigen::half h = Eigen::half(1.0f);
    gamma_tensor->add_half_val(*reinterpret_cast<const uint16*>(&h));
  } else if (dtype == DT_BFLOAT16) {
    bfloat16 bf = bfloat16(1.0f);
    gamma_tensor->add_half_val(*reinterpret_cast<const uint16*>(&bf));
  }
  // 设置为标量 (rank-0)
  gamma_tensor->mutable_tensor_shape()->Clear();  // Empty shape = scalar

  // 创建 beta (默认为标量 0.0)
  NodeDef* beta_node = graph->add_node();
  beta_node->set_name(output_name + "/beta");
  beta_node->set_op("Const");
  beta_node->set_device(output_device);
  auto* beta_attr = beta_node->mutable_attr();
  (*beta_attr)["dtype"].set_type(dtype);
  TensorProto* beta_tensor = (*beta_attr)["value"].mutable_tensor();
  beta_tensor->set_dtype(dtype);
  // 设置标量值 0.0
  if (dtype == DT_FLOAT) {
    beta_tensor->add_float_val(0.0f);
  } else if (dtype == DT_HALF) {
    Eigen::half h = Eigen::half(0.0f);
    beta_tensor->add_half_val(*reinterpret_cast<const uint16*>(&h));
  } else if (dtype == DT_BFLOAT16) {
    bfloat16 bf = bfloat16(0.0f);
    beta_tensor->add_half_val(*reinterpret_cast<const uint16*>(&bf));
  }
  // 设置为标量 (rank-0)
  beta_tensor->mutable_tensor_shape()->Clear();  // Empty shape = scalar

  // 创建融合节点
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(output_name);
  fused_node->set_op("MusaNormalize");
  fused_node->set_device(output_device);

  fused_node->add_input(input_name);
  fused_node->add_input(gamma_node->name());
  fused_node->add_input(beta_node->name());

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["epsilon"].set_f(epsilon);
  (*attr)["max_std"].set_f(max_std);

  VLOG(1) << "[Normalize::Apply] SUCCESS fused to " << output_name
          << ", reduction_indices=" << IndicesToString(reduction_indices)
          << ", epsilon=" << epsilon << ", max_std=" << max_std
          << ", removed=" << removed_count
          << ", graph_nodes=" << graph->node_size();

  return Status::OK();
}

// 注册融合模式
REGISTER_FUSION_PATTERN(MusaNormalizeFusion);

// 注册 kernel 可用性
REGISTER_FUSION_KERNEL(MusaNormalizeFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
