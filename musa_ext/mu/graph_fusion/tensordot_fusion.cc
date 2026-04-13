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

#include "mu/graph_fusion/tensordot_fusion.h"

#include <cstring>
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

// 权重节点的有效类型
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

// Helper to find TensorDot prefix from node name.
// Reshape_2 的名字本身就是 prefix, 如 ".../Tensordot"
// 内部子节点名字形如 prefix + "/xx"
std::string FindTensorDotPrefix(const std::string& node_name) {
  size_t pos = node_name.rfind("/Tensordot");
  if (pos == std::string::npos) {
    pos = node_name.rfind("/tensordot");
  }
  if (pos != std::string::npos) {
    return node_name.substr(0, pos + strlen("/Tensordot"));
  }
  return "";
}

// 判断节点是否属于同一个 Tensordot 子图。
// 规则: node_name == prefix (Reshape_2自身) 或 node_name 以 prefix+"/" 开头。
bool BelongsToTensordot(const std::string& node_name,
                        const std::string& prefix) {
  if (prefix.empty()) return false;
  if (node_name == prefix) return true;
  return node_name.length() > prefix.length() &&
         node_name.compare(0, prefix.length(), prefix) == 0 &&
         node_name[prefix.length()] == '/';
}

// Helper to collect all nodes with a given prefix
std::vector<std::string> CollectNodesWithPrefix(const GraphDef& graph,
                                                const std::string& prefix) {
  std::vector<std::string> nodes;
  if (prefix.empty()) return nodes;

  for (int i = 0; i < graph.node_size(); ++i) {
    const std::string& name = graph.node(i).name();
    // Check if name starts with prefix followed by '/' or equals prefix
    if (name == prefix || (name.length() > prefix.length() &&
                           name.substr(0, prefix.length()) == prefix &&
                           name[prefix.length()] == '/')) {
      nodes.push_back(name);
    }
  }
  return nodes;
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
// Returns final Const node or nullptr if pattern not satisfied.
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

// Helper: check that an input comes from a specific op type (optionally
// through Identity). Returns the first node with the expected op or nullptr.
const NodeDef* GetOpLikeNode(const GraphDef& graph,
                             const std::string& input_name,
                             const std::string& target_op) {
  const NodeDef* node = FindProducer(graph, input_name);
  if (!node) return nullptr;
  while (node && IsOp(*node, "Identity")) {
    if (node->input_size() == 0) return nullptr;
    node = FindProducer(graph, node->input(0));
  }
  if (!node || !IsOp(*node, target_op)) return nullptr;
  return node;
}

// Extract axes indices from GatherV2's indices constant.
// Returns empty vector if extraction fails.
std::vector<int> ExtractAxesFromGather(const GraphDef& graph,
                                       const NodeDef* gather_node) {
  std::vector<int> axes;
  if (!gather_node || gather_node->input_size() < 2) return axes;

  const NodeDef* indices_const = GetConstLikeNode(graph, gather_node->input(1));
  if (!indices_const) return axes;

  auto value_it = indices_const->attr().find("value");
  if (value_it == indices_const->attr().end()) return axes;

  const TensorProto& tp = value_it->second.tensor();

  if (tp.dtype() == DT_INT32) {
    if (tp.int_val_size() > 0) {
      for (int i = 0; i < tp.int_val_size(); ++i) {
        axes.push_back(tp.int_val(i));
      }
    } else if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int32_t);
      const int32_t* data =
          reinterpret_cast<const int32_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        axes.push_back(data[i]);
      }
    }
  } else if (tp.dtype() == DT_INT64) {
    if (tp.int64_val_size() > 0) {
      for (int i = 0; i < tp.int64_val_size(); ++i) {
        axes.push_back(static_cast<int>(tp.int64_val(i)));
      }
    } else if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int64_t);
      const int64_t* data =
          reinterpret_cast<const int64_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        axes.push_back(static_cast<int>(data[i]));
      }
    }
  }

  // Scalar tensor (rank-0): stored as single int_val
  if (axes.empty() && tp.tensor_shape().dim_size() == 0) {
    if (tp.dtype() == DT_INT32 && tp.int_val_size() == 1) {
      axes.push_back(tp.int_val(0));
    } else if (tp.dtype() == DT_INT64 && tp.int64_val_size() == 1) {
      axes.push_back(static_cast<int>(tp.int64_val(0)));
    }
  }

  return axes;
}

// Serialize axes vector to comma-separated string for captured_attrs.
std::string AxesToString(const std::vector<int>& axes) {
  std::ostringstream oss;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i > 0) oss << ",";
    oss << axes[i];
  }
  return oss.str();
}

// Parse comma-separated axes string back to vector.
std::vector<int> StringToAxes(const std::string& s) {
  std::vector<int> axes;
  if (s.empty()) return axes;
  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    axes.push_back(std::stoi(token));
  }
  return axes;
}

}  // namespace

// =============================================================================
// MusaTensorDotFusion Implementation
// =============================================================================

MusaTensorDotFusion::MusaTensorDotFusion() = default;

bool MusaTensorDotFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaTensorDotFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    VLOG(2) << "[TensorDot::Match] RETURN empty: node_idx out of range";
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  if (IsOp(start_node, "Reshape")) {
    VLOG(2) << "[TensorDot::Match] ENTER Reshape path, node="
            << start_node.name();
    return MatchFromReshapeNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaTensorDotFusion::MatchFromReshapeNode(
    const GraphDef& graph, int reshape_node_idx) const {
  FusionMatchResult result;
  const NodeDef& reshape_2 = graph.node(reshape_node_idx);

  VLOG(2) << "[TensorDot::Match] MatchFromReshapeNode ENTER, node="
          << reshape_2.name();

  if (!IsOp(reshape_2, "Reshape")) {
    VLOG(2) << "[TensorDot::Match] FAIL: not Reshape op, node="
            << reshape_2.name();
    return result;
  }

  // =========================================================================
  // 前置: 提取 Tensordot 前缀。
  // Reshape_2 本身的名字就是前缀, 如 ".../Tensordot"
  // 所有内部节点命名为 prefix + "/xxx"
  // =========================================================================
  const std::string tensordot_prefix = FindTensorDotPrefix(reshape_2.name());
  if (tensordot_prefix.empty()) {
    VLOG(2) << "[TensorDot::Match] FAIL: cannot extract Tensordot prefix, node="
            << reshape_2.name();
    return result;
  }
  if (reshape_2.name() != tensordot_prefix) {
    VLOG(2)
        << "[TensorDot::Match] FAIL: Reshape_2 name != tensordot_prefix, name="
        << reshape_2.name() << ", prefix=" << tensordot_prefix;
    return result;
  }
  VLOG(2) << "[TensorDot::Match] prefix=" << tensordot_prefix;

  // 为快速匹配构建 node map
  auto node_map = BuildNodeMap(graph);

  // =========================================================================
  // 第1步: Reshape_2 的两个输入必须是:
  //   - tensor: MatMul   (必须属于同一前缀)
  //   - shape:  ConcatV2 (必须属于同一前缀)
  // =========================================================================
  if (reshape_2.input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step1: Reshape_2 input_size="
            << reshape_2.input_size() << " (need 2), node=" << reshape_2.name();
    return result;
  }

  const NodeDef* matmul_node = FindProducer(graph, reshape_2.input(0));
  const NodeDef* concat_node = FindProducer(graph, reshape_2.input(1));

  if (!matmul_node || !IsOp(*matmul_node, "MatMul")) {
    VLOG(2) << "[TensorDot::Match] FAIL step1: input[0] is not MatMul, actual="
            << (matmul_node ? matmul_node->op() : "NULL")
            << ", node=" << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(matmul_node->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step1: MatMul not in prefix, name="
            << matmul_node->name() << ", prefix=" << tensordot_prefix;
    return result;
  }
  if (!concat_node || !IsOp(*concat_node, "ConcatV2")) {
    VLOG(2)
        << "[TensorDot::Match] FAIL step1: input[1] is not ConcatV2, actual="
        << (concat_node ? concat_node->op() : "NULL")
        << ", node=" << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(concat_node->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step1: ConcatV2 not in prefix, name="
            << concat_node->name() << ", prefix=" << tensordot_prefix;
    return result;
  }

  VLOG(2) << "[TensorDot::Match] PASS step1: MatMul=" << matmul_node->name()
          << ", ConcatV2=" << concat_node->name();

  // =========================================================================
  // 第2步: MatMul 的输入是 Reshape_1(属于前缀) + 权重
  // =========================================================================
  if (matmul_node->input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step2: MatMul input_size="
            << matmul_node->input_size() << ", node=" << reshape_2.name();
    return result;
  }

  const NodeDef* input_a = FindProducer(graph, matmul_node->input(0));
  const NodeDef* input_b = FindProducer(graph, matmul_node->input(1));
  if (!input_a || !input_b) {
    VLOG(2) << "[TensorDot::Match] FAIL step2: cannot find MatMul inputs, node="
            << reshape_2.name();
    return result;
  }

  const NodeDef* reshape_1 = nullptr;
  const NodeDef* weight_node = nullptr;

  if (IsOp(*input_a, "Reshape") && IsWeightOp(*input_b)) {
    reshape_1 = input_a;
    weight_node = input_b;
  } else {
    VLOG(2) << "[TensorDot::Match] FAIL step2: MatMul inputs mismatch, a.op="
            << input_a->op() << ", b.op=" << input_b->op()
            << ", node=" << reshape_2.name();
    return result;
  }

  if (!BelongsToTensordot(reshape_1->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step2: Reshape_1 not in prefix, name="
            << reshape_1->name() << ", prefix=" << tensordot_prefix;
    return result;
  }

  VLOG(2) << "[TensorDot::Match] PASS step2: Reshape_1=" << reshape_1->name()
          << ", Weight=" << weight_node->name() << "(" << weight_node->op()
          << ")";

  // =========================================================================
  // 第3步: Reshape_1 的输入:
  //   - tensor: 任意 op (可能是 Transpose/Placeholder/Mul/AddV2 等)
  //   - shape:  Pack     (必须属于前缀)
  //   如果 tensor 输入是 Transpose 且属于前缀，它是 Tensordot 内部 Transpose;
  //   否则是外部节点或被 Grappler 优化后的替代。
  // =========================================================================
  const NodeDef* transpose_node = nullptr;
  const NodeDef* pack_node = nullptr;
  std::string original_input_name;

  if (reshape_1->input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step3: Reshape_1 input_size="
            << reshape_1->input_size() << ", node=" << reshape_2.name();
    return result;
  }

  const NodeDef* r_in0 = FindProducer(graph, reshape_1->input(0));
  const NodeDef* r_in1 = FindProducer(graph, reshape_1->input(1));
  if (!r_in0 || !r_in1) {
    VLOG(2)
        << "[TensorDot::Match] FAIL step3: cannot find Reshape_1 inputs, node="
        << reshape_2.name();
    return result;
  }

  // 识别 Pack 和 tensor 输入
  const NodeDef* tensor_input_node = nullptr;
  int tensor_input_idx = -1;  // reshape_1->input 中 tensor 侧的索引
  if (IsOp(*r_in0, "Pack") &&
      BelongsToTensordot(r_in0->name(), tensordot_prefix)) {
    pack_node = r_in0;
    tensor_input_node = r_in1;
    tensor_input_idx = 1;
  } else if (IsOp(*r_in1, "Pack") &&
             BelongsToTensordot(r_in1->name(), tensordot_prefix)) {
    pack_node = r_in1;
    tensor_input_node = r_in0;
    tensor_input_idx = 0;
  } else {
    VLOG(2) << "[TensorDot::Match] FAIL step3: no Pack in prefix found, "
            << "in0=" << r_in0->name() << "(" << r_in0->op() << "), "
            << "in1=" << r_in1->name() << "(" << r_in1->op() << "), "
            << "node=" << reshape_2.name();
    return result;
  }

  // 判断 tensor 输入: 如果是 Transpose 且属于前缀 → 内部 Transpose
  bool transpose_is_internal = false;
  if (IsOp(*tensor_input_node, "Transpose") &&
      BelongsToTensordot(tensor_input_node->name(), tensordot_prefix)) {
    transpose_node = tensor_input_node;
    transpose_is_internal = true;
    if (transpose_node->input_size() > 0) {
      original_input_name = GetCleanName(transpose_node->input(0));
    }
    VLOG(2) << "[TensorDot::Match] step3: internal Transpose="
            << transpose_node->name() << ", input_a=" << original_input_name;
  } else {
    original_input_name = GetCleanName(reshape_1->input(tensor_input_idx));
    VLOG(2) << "[TensorDot::Match] step3: tensor_input=" << original_input_name
            << " (op=" << tensor_input_node->op() << ")";
  }

  VLOG(2) << "[TensorDot::Match] PASS step3: input_a=" << original_input_name
          << ", Pack=" << pack_node->name() << ", node=" << reshape_2.name();

  // =========================================================================
  // 第4步: Pack → Prod_1/Prod_2 → GatherV2_1/GatherV2_2 → Shape_1
  //        所有这些节点都必须属于前缀（但 Const 类型的可以是共享的）
  // =========================================================================
  if (pack_node->input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Pack input_size="
            << pack_node->input_size() << ", node=" << reshape_2.name();
    return result;
  }

  const NodeDef* prod_1 = FindProducer(graph, pack_node->input(0));
  const NodeDef* prod_2 = FindProducer(graph, pack_node->input(1));
  if (!prod_1 || !prod_2 || !IsOp(*prod_1, "Prod") || !IsOp(*prod_2, "Prod")) {
    VLOG(2)
        << "[TensorDot::Match] FAIL step4: Pack inputs not both Prod, in0.op="
        << (prod_1 ? prod_1->op() : "NULL")
        << ", in1.op=" << (prod_2 ? prod_2->op() : "NULL")
        << ", node=" << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(prod_1->name(), tensordot_prefix) ||
      !BelongsToTensordot(prod_2->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Prod not in prefix, "
            << "prod_1=" << prod_1->name() << ", prod_2=" << prod_2->name();
    return result;
  }

  // Prod_1 → GatherV2_1
  if (prod_1->input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Prod_1 input_size="
            << prod_1->input_size() << ", node=" << reshape_2.name();
    return result;
  }
  const NodeDef* gather_1 = GetOpLikeNode(graph, prod_1->input(0), "GatherV2");
  const NodeDef* prod1_red = GetConstLikeNode(graph, prod_1->input(1));
  if (!gather_1 || !prod1_red) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Prod_1 pattern mismatch, node="
            << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(gather_1->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2_1 not in prefix, name="
            << gather_1->name();
    return result;
  }

  // Prod_2 → GatherV2_2
  if (prod_2->input_size() != 2) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Prod_2 input_size="
            << prod_2->input_size() << ", node=" << reshape_2.name();
    return result;
  }
  const NodeDef* gather_2 = GetOpLikeNode(graph, prod_2->input(0), "GatherV2");
  const NodeDef* prod2_red = GetConstLikeNode(graph, prod_2->input(1));
  if (!gather_2 || !prod2_red) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Prod_2 pattern mismatch, node="
            << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(gather_2->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2_2 not in prefix, name="
            << gather_2->name();
    return result;
  }

  VLOG(2) << "[TensorDot::Match] PASS step4: Prod_1=" << prod_1->name()
          << ", Prod_2=" << prod_2->name()
          << ", GatherV2_1=" << gather_1->name()
          << ", GatherV2_2=" << gather_2->name();

  // GatherV2_1 / GatherV2_2 的 params 来自同一个 Shape_1（属于前缀）
  const NodeDef* shape_1 = GetOpLikeNode(graph, gather_1->input(0), "Shape");
  if (!shape_1) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2_1 params is not Shape, "
               "node="
            << reshape_2.name();
    return result;
  }
  if (!BelongsToTensordot(shape_1->name(), tensordot_prefix)) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: Shape_1 not in prefix, name="
            << shape_1->name();
    return result;
  }
  const NodeDef* shape_1_b = GetOpLikeNode(graph, gather_2->input(0), "Shape");
  if (!shape_1_b || shape_1_b != shape_1) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2_2 params not same "
               "Shape_1, shape_1="
            << shape_1->name()
            << ", shape_1_b=" << (shape_1_b ? shape_1_b->name() : "NULL")
            << ", node=" << reshape_2.name();
    return result;
  }

  if (gather_1->input_size() < 3 || gather_2->input_size() < 3) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2 input_size < 3, node="
            << reshape_2.name();
    return result;
  }

  if (!GetConstLikeNode(graph, gather_1->input(1)) ||
      !GetConstLikeNode(graph, gather_1->input(2)) ||
      !GetConstLikeNode(graph, gather_2->input(1)) ||
      !GetConstLikeNode(graph, gather_2->input(2))) {
    VLOG(2) << "[TensorDot::Match] FAIL step4: GatherV2 indices/axis not "
               "const-like, node="
            << reshape_2.name();
    return result;
  }

  VLOG(2) << "[TensorDot::Match] PASS step4: Shape_1=" << shape_1->name()
          << ", GatherV2 indices/axis all const-like";

  // =========================================================================
  // 第5步: ConcatV2（已验证属于前缀）至少包含 GatherV2_1，其余 Const；axis
  // Const
  // =========================================================================
  if (concat_node->input_size() < 3) {
    VLOG(2) << "[TensorDot::Match] FAIL step5: ConcatV2 input_size="
            << concat_node->input_size()
            << " (need >=3), node=" << reshape_2.name();
    return result;
  }

  const NodeDef* concat_axis = GetConstLikeNode(
      graph, concat_node->input(concat_node->input_size() - 1));
  if (!concat_axis) {
    VLOG(2)
        << "[TensorDot::Match] FAIL step5: ConcatV2 axis not const-like, node="
        << reshape_2.name();
    return result;
  }

  bool has_gather1_in_concat = false;
  for (int i = 0; i < concat_node->input_size() - 1; ++i) {
    const NodeDef* v = FindProducer(graph, concat_node->input(i));
    if (!v) continue;

    const NodeDef* base_v = v;
    while (base_v && IsOp(*base_v, "Identity")) {
      if (base_v->input_size() == 0) break;
      base_v = FindProducer(graph, base_v->input(0));
    }

    if (base_v == gather_1) {
      has_gather1_in_concat = true;
      continue;
    }

    if (!GetConstLikeNode(graph, concat_node->input(i))) {
      VLOG(2) << "[TensorDot::Match] FAIL step5: ConcatV2 value[" << i
              << "] not GatherV2_1 or const, node=" << reshape_2.name();
      return result;
    }
  }

  if (!has_gather1_in_concat) {
    VLOG(2)
        << "[TensorDot::Match] FAIL step5: ConcatV2 missing GatherV2_1, node="
        << reshape_2.name();
    return result;
  }

  VLOG(2) << "[TensorDot::Match] PASS step5: ConcatV2=" << concat_node->name()
          << " contains GatherV2_1, axis is const";

  // =========================================================================
  // 从 GatherV2 提取 axes
  // gather_2 (from Prod_2 → Pack.input(1)) 对应 A 的收缩轴 axes_a
  // gather_1 (from Prod_1 → Pack.input(0)) 对应 A 的自由轴 (free dims)
  // axes_b 默认为 [0]（权重的收缩轴通常在第一维）
  // =========================================================================
  std::vector<int> axes_a = ExtractAxesFromGather(graph, gather_2);
  if (axes_a.empty()) {
    axes_a.push_back(-1);
    LOG(WARNING) << "[TensorDot::Match] could not extract axes_a from "
                 << "GatherV2, using default [-1], node=" << reshape_2.name();
  }
  std::vector<int> axes_b;
  axes_b.push_back(0);

  VLOG(2) << "[TensorDot::Match] extracted axes_a=" << AxesToString(axes_a)
          << ", axes_b=" << AxesToString(axes_b)
          << ", node=" << reshape_2.name();

  // =========================================================================
  // 收集所有需要融合的节点
  // =========================================================================
  std::vector<std::string> nodes_to_fuse =
      CollectNodesWithPrefix(graph, tensordot_prefix);

  // =========================================================================
  // 构建匹配结果
  // =========================================================================
  result.matched = true;

  result.matched_nodes.push_back(&reshape_2);
  result.matched_nodes.push_back(matmul_node);
  result.matched_nodes.push_back(reshape_1);
  result.matched_nodes.push_back(concat_node);
  if (pack_node) {
    result.matched_nodes.push_back(pack_node);
  }

  result.captured_nodes["output"] = &reshape_2;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["reshape_1"] = reshape_1;
  result.captured_nodes["concat"] = concat_node;
  result.captured_nodes["weight"] = weight_node;
  if (pack_node) {
    result.captured_nodes["pack"] = pack_node;
  }

  result.captured_attrs["original_input"] = original_input_name;
  result.captured_attrs["weight_input"] = weight_node->name();
  result.captured_attrs["tensordot_prefix"] = tensordot_prefix;
  result.captured_attrs["axes_a"] = AxesToString(axes_a);
  result.captured_attrs["axes_b"] = AxesToString(axes_b);

  for (const auto& node_name : nodes_to_fuse) {
    result.captured_attrs["fuse_node_" +
                          std::to_string(result.captured_attrs.size())] =
        node_name;
  }

  VLOG(1) << "[TensorDot::Match] SUCCESS matched=" << reshape_2.name()
          << ", input_a=" << original_input_name
          << ", weight=" << weight_node->name()
          << ", axes_a=" << AxesToString(axes_a)
          << ", axes_b=" << AxesToString(axes_b)
          << ", prefix=" << tensordot_prefix
          << ", fuse_nodes=" << nodes_to_fuse.size();

  return result;
}

Status MusaTensorDotFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  VLOG(2) << "[TensorDot::Apply] ENTER, matched=" << match_result.matched
          << ", nodes_count=" << match_result.matched_nodes.size()
          << ", kernel_available=" << IsKernelAvailable();

  if (!match_result.IsValid()) {
    VLOG(2) << "[TensorDot::Apply] RETURN: invalid match result";
    return ::tsl::errors::InvalidArgument("Invalid TensorDot match result");
  }

  if (!IsKernelAvailable()) {
    VLOG(2)
        << "[TensorDot::Apply] RETURN: kernel not available, skipping fusion";
    return ::tsl::OkStatus();
  }

  // 获取关键节点
  auto output_it = match_result.captured_nodes.find("output");
  auto weight_it = match_result.captured_nodes.find("weight");

  if (output_it == match_result.captured_nodes.end()) {
    VLOG(2)
        << "[TensorDot::Apply] RETURN: missing output node in captured_nodes";
    return ::tsl::errors::InvalidArgument(
                  "Missing output node in TensorDot pattern");
  }

  const NodeDef* output_node = output_it->second;
  std::string output_name = output_node->name();
  std::string output_device = output_node->device();
  VLOG(2) << "[TensorDot::Apply] output_node=" << output_name;

  // 检查是否已经融合过
  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaTensorDot") {
      VLOG(2) << "[TensorDot::Apply] RETURN: already fused, node="
              << output_name;
      return ::tsl::errors::AlreadyExists("Already fused");
    }
  }

  // 获取输入名称
  std::string input_a_name;
  std::string input_b_name;

  auto original_input_it = match_result.captured_attrs.find("original_input");
  if (original_input_it != match_result.captured_attrs.end() &&
      !original_input_it->second.empty()) {
    input_a_name = original_input_it->second;
  } else {
    VLOG(2) << "[TensorDot::Apply] RETURN: cannot determine input A";
    return ::tsl::errors::InvalidArgument(
                  "Cannot determine TensorDot input A");
  }

  auto weight_input_it = match_result.captured_attrs.find("weight_input");
  if (weight_input_it != match_result.captured_attrs.end() &&
      !weight_input_it->second.empty()) {
    input_b_name = weight_input_it->second;
  } else if (weight_it != match_result.captured_nodes.end() &&
             weight_it->second) {
    input_b_name = weight_it->second->name();
  } else {
    VLOG(2) << "[TensorDot::Apply] RETURN: cannot determine input B (weight)";
    return ::tsl::errors::InvalidArgument(
                  "Cannot determine TensorDot input B (weight)");
  }

  // 获取数据类型
  DataType dtype = DT_FLOAT;
  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    dtype = dtype_it->second.type();
  }

  // 提取 axes
  std::vector<int> axes_a, axes_b;
  auto axes_a_it = match_result.captured_attrs.find("axes_a");
  if (axes_a_it != match_result.captured_attrs.end()) {
    axes_a = StringToAxes(axes_a_it->second);
  }
  if (axes_a.empty()) axes_a.push_back(-1);

  auto axes_b_it = match_result.captured_attrs.find("axes_b");
  if (axes_b_it != match_result.captured_attrs.end()) {
    axes_b = StringToAxes(axes_b_it->second);
  }
  if (axes_b.empty()) axes_b.push_back(0);

  VLOG(2) << "[TensorDot::Apply] input_a=" << input_a_name
          << ", input_b=" << input_b_name << ", axes_a=" << AxesToString(axes_a)
          << ", axes_b=" << AxesToString(axes_b);

  // =========================================================================
  // 收集需要删除的节点名称（从 captured_attrs 中的 fuse_node_* 提取）
  // =========================================================================
  std::unordered_set<std::string> fuse_node_names;
  for (const auto& kv : match_result.captured_attrs) {
    if (kv.first.substr(0, 10) == "fuse_node_") {
      fuse_node_names.insert(kv.second);
    }
  }

  fuse_node_names.erase(input_a_name);
  fuse_node_names.erase(input_b_name);

  // 保留被子图外部引用的内部节点
  std::unordered_set<std::string> shared_nodes;
  for (int i = 0; i < graph->node_size(); ++i) {
    const NodeDef& node = graph->node(i);
    if (fuse_node_names.count(node.name())) continue;

    for (int j = 0; j < node.input_size(); ++j) {
      std::string producer =
          FusionGraphUtils::GetProducerNodeName(node.input(j));
      if (fuse_node_names.count(producer) && producer != output_name) {
        shared_nodes.insert(producer);
      }
    }
  }
  for (const auto& name : shared_nodes) {
    VLOG(2) << "[TensorDot::Apply] keeping shared node: " << name;
    fuse_node_names.erase(name);
  }

  VLOG(2) << "[TensorDot::Apply] will remove " << fuse_node_names.size()
          << " fused sub-graph nodes";

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
            producer != input_a_name && producer != input_b_name) {
          potential_orphan_producers.insert(producer);
        }
      }
    }
  }

  VLOG(2) << "[TensorDot::Apply] found " << potential_orphan_producers.size()
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

  VLOG(2) << "[TensorDot::Apply] removed " << removed_count << " nodes";

  // =========================================================================
  // 检查并删除孤立节点
  // =========================================================================
  int orphan_removed_count = 0;
  for (auto it = potential_orphan_producers.begin();
       it != potential_orphan_producers.end();) {
    const std::string& producer_name = *it;

    // 检查该节点是否还存在（可能已经被之前的删除操作删除）
    int idx = FusionGraphUtils::FindNodeIndex(*graph, producer_name);
    if (idx < 0 || idx >= graph->node_size()) {
      ++it;
      continue;
    }

    // 检查该节点是否还有其他消费者
    bool has_consumers = false;
    for (int i = 0; i < graph->node_size(); ++i) {
      const NodeDef& node = graph->node(i);
      for (int j = 0; j < node.input_size(); ++j) {
        std::string input_producer =
            FusionGraphUtils::GetProducerNodeName(node.input(j));
        if (input_producer == producer_name) {
          has_consumers = true;
          break;
        }
      }
      if (has_consumers) break;
    }

    if (!has_consumers) {
      // 孤立节点，删除它
      VLOG(2) << "[TensorDot::Apply] removing orphan node: " << producer_name;
      FusionGraphUtils::RemoveNode(graph, idx);
      orphan_removed_count++;
      it = potential_orphan_producers.erase(it);
    } else {
      ++it;
    }
  }

  if (orphan_removed_count > 0) {
    VLOG(2) << "[TensorDot::Apply] removed " << orphan_removed_count
            << " orphan nodes";
  }

  // =========================================================================
  // 创建融合节点
  // =========================================================================
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(output_name);
  fused_node->set_op("MusaTensorDot");
  fused_node->set_device(output_device);

  fused_node->add_input(input_a_name);
  fused_node->add_input(input_b_name);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"].set_type(dtype);

  auto* axes_a_list = (*attr)["axes_a"].mutable_list();
  for (int a : axes_a) axes_a_list->add_i(a);

  auto* axes_b_list = (*attr)["axes_b"].mutable_list();
  for (int b : axes_b) axes_b_list->add_i(b);

  VLOG(1) << "[TensorDot::Apply] SUCCESS fused to " << output_name
          << ", axes_a=" << AxesToString(axes_a)
          << ", axes_b=" << AxesToString(axes_b)
          << ", removed=" << removed_count
          << ", graph_nodes=" << graph->node_size();

  return ::tsl::OkStatus();
}

// 注册融合模式
REGISTER_FUSION_PATTERN(MusaTensorDotFusion);

// 注册 kernel 可用性
REGISTER_FUSION_KERNEL(MusaTensorDotFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
