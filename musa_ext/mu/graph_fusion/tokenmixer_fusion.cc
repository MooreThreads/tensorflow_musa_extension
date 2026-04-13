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

/// fusion pattern: reshape ([-1, T, H, d_k]) -> transpose (perm=[0, 2, 1, 3])
/// -> reshape ([-1, H, T*d_k])
#include "mu/graph_fusion/tokenmixer_fusion.h"

#include "fusion_pattern_manager.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {
namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
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

bool GetConstIntValues(const NodeDef& node, std::vector<int64>* values) {
  if (!IsOp(node, "Const")) return false;

  auto it = node.attr().find("value");
  if (it == node.attr().end()) return false;

  const auto& tensor_proto = it->second.tensor();

  if (tensor_proto.dtype() == DT_INT32) {
    if (tensor_proto.int_val_size() > 0) {
      for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
        values->push_back(static_cast<int64>(tensor_proto.int_val(i)));
      }
    } else if (!tensor_proto.tensor_content().empty()) {
      const int32* data =
          reinterpret_cast<const int32*>(tensor_proto.tensor_content().data());
      int num = tensor_proto.tensor_content().size() / sizeof(int32);
      for (int i = 0; i < num; ++i) {
        values->push_back(static_cast<int64>(data[i]));
      }
    }
  } else if (tensor_proto.dtype() == DT_INT64) {
    if (tensor_proto.int64_val_size() > 0) {
      for (int i = 0; i < tensor_proto.int64_val_size(); ++i) {
        values->push_back(tensor_proto.int64_val(i));
      }
    } else if (!tensor_proto.tensor_content().empty()) {
      const int64* data =
          reinterpret_cast<const int64*>(tensor_proto.tensor_content().data());
      int num = tensor_proto.tensor_content().size() / sizeof(int64);
      for (int i = 0; i < num; ++i) {
        values->push_back(data[i]);
      }
    }
  } else {
    return false;
  }

  return !values->empty();
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

}  // namespace

// =============================================================================
// MusaTokenMixerFusion Implementation
// =============================================================================

MusaTokenMixerFusion::MusaTokenMixerFusion() = default;

bool MusaTokenMixerFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaTokenMixerFusion::Match(const GraphDef& graph,
                                              int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start_node = graph.node(start_node_idx);

  if (IsOp(start_node, "Reshape")) {
    return MatchFromReshapeNode(graph, start_node_idx);
  }

  return FusionMatchResult{};
}

FusionMatchResult MusaTokenMixerFusion::MatchFromReshapeNode(
    const GraphDef& graph, int reshape_node_idx) const {
  FusionMatchResult result;
  const NodeDef& reshape2_node = graph.node(reshape_node_idx);

  if (!IsOp(reshape2_node, "Reshape") || reshape2_node.input_size() < 2) {
    printf(
        "MusaTokenMixerFusion: MatchFromReshapeNode: Invalid Reshape node\n");
    return result;
  }
  // Step 0: Output Reshape must have downstream consumers
  const std::string& name = reshape2_node.name();
  if (name.size() > 9 && name.substr(name.size() - 9) == "_original") {
    VLOG(2) << "MusaTokenMixer: Node '" << name << "' already fused, skipping";
    return result;
  }
  // =========================================================================
  // Step 1: The last Reshape's data input (input[0]) must be a Transpose
  // =========================================================================
  const NodeDef* transpose_node = FindProducer(graph, reshape2_node.input(0));
  if (!transpose_node || !IsOp(*transpose_node, "Transpose")) {
    return result;
  }

  // =========================================================================
  // Step 2: Validate Transpose perm = [0, 2, 1, 3]
  //   Case A: perm stored as a list(int) attribute on the Transpose node
  //   Case B: perm stored as a separate Const input node (standard TF graph)
  // =========================================================================
  std::vector<int64> perm_values;

  auto perm_attr_it = transpose_node->attr().find("perm");
  if (perm_attr_it != transpose_node->attr().end()) {
    const auto& perm_list = perm_attr_it->second.list();
    for (int i = 0; i < perm_list.i_size(); ++i) {
      perm_values.push_back(perm_list.i(i));
    }
  }

  if (perm_values.empty() && transpose_node->input_size() >= 2) {
    const NodeDef* perm_node = FindProducer(graph, transpose_node->input(1));
    if (perm_node) {
      GetConstIntValues(*perm_node, &perm_values);
    }
  }

  if (perm_values.size() != 4 || perm_values[0] != 0 || perm_values[1] != 2 ||
      perm_values[2] != 1 || perm_values[3] != 3) {
    return result;
  }

  VLOG(2) << "MusaTokenMixer: Found Transpose with perm=[0,2,1,3]";

  // =========================================================================
  // Step 3: Transpose's data input (input[0]) must be the first Reshape
  // =========================================================================
  if (transpose_node->input_size() < 1) return result;

  const NodeDef* reshape1_node = FindProducer(graph, transpose_node->input(0));
  if (!reshape1_node || !IsOp(*reshape1_node, "Reshape")) {
    return result;
  }

  if (reshape1_node->input_size() < 2) {
    printf(
        "MusaTokenMixerFusion: MatchFromReshapeNode: Invalid Reshape node\n");
    return result;
  }
  // =========================================================================
  // Step 4: Intermediate nodes must be single-consumer
  // =========================================================================
  int reshape1_consumers = CountConsumers(graph, reshape1_node->name());
  int transpose_consumers = CountConsumers(graph, transpose_node->name());

  if (reshape1_consumers != 1 || transpose_consumers != 1) {
    VLOG(2) << "MusaTokenMixer: Intermediate nodes have multiple consumers, "
            << "skipping (reshape1=" << reshape1_consumers
            << ", transpose=" << transpose_consumers << ")";
    return result;
  }

  // =========================================================================
  // Step 5: Extract shape from the first Reshape -> expect [-1, T, H, d_k]
  // =========================================================================
  const NodeDef* shape1_node = FindProducer(graph, reshape1_node->input(1));
  if (!shape1_node) return result;

  std::vector<int64> shape1_values;
  if (!GetConstIntValues(*shape1_node, &shape1_values)) {
    return result;
  }

  if (shape1_values.size() != 4) {
    return result;
  }

  if (shape1_values[0] != -1) {
    VLOG(2) << "MusaTokenMixer: First Reshape batch dim is not -1, got "
            << shape1_values[0];
    return result;
  }

  int64 num_T = shape1_values[1];
  int64 num_H = shape1_values[2];
  int64 d_k = shape1_values[3];

  if (num_T <= 0 || num_H <= 0 || d_k <= 0) {
    return result;
  }

  VLOG(2) << "MusaTokenMixer: First Reshape shape=[" << shape1_values[0] << ", "
          << num_T << ", " << num_H << ", " << d_k << "]";

  // =========================================================================
  // Step 6: Validate the last Reshape's shape -> expect [-1, H, T*d_k]
  // =========================================================================
  const NodeDef* shape2_node = FindProducer(graph, reshape2_node.input(1));
  if (!shape2_node) return result;

  std::vector<int64> shape2_values;
  if (!GetConstIntValues(*shape2_node, &shape2_values)) {
    return result;
  }

  if (shape2_values.size() != 3) {
    return result;
  }

  if (shape2_values[0] != -1) {
    VLOG(2) << "MusaTokenMixer: Last Reshape batch dim is not -1, got "
            << shape2_values[0];
    return result;
  }

  if (shape2_values[1] != num_H || shape2_values[2] != num_T * d_k) {
    VLOG(2) << "MusaTokenMixer: Last Reshape shape mismatch. "
            << "Expected [-1, " << num_H << ", " << num_T * d_k << "], "
            << "got [" << shape2_values[0] << ", " << shape2_values[1] << ", "
            << shape2_values[2] << "]";
    return result;
  }

  // =========================================================================
  // Step 7: Validate dtype consistency across the chain
  // =========================================================================
  DataType dt_reshape1 = GetNodeDType(*reshape1_node);
  DataType dt_transpose = GetNodeDType(*transpose_node);
  DataType dt_reshape2 = GetNodeDType(reshape2_node);

  if (dt_reshape1 != DT_INVALID && dt_transpose != DT_INVALID &&
      dt_reshape2 != DT_INVALID) {
    if (dt_reshape1 != dt_transpose || dt_transpose != dt_reshape2) {
      VLOG(2) << "MusaTokenMixer: dtype mismatch across chain";
      return result;
    }
  }
  // =========================================================================
  // Step 8: Validate first Reshape's input is 3D with D == H * d_k
  // =========================================================================
  const NodeDef* input_node = FindProducer(graph, reshape1_node->input(0));
  if (!input_node) return result;

  // Check _output_shapes attribute (present in most frozen/optimized graphs)
  auto shapes_it = input_node->attr().find("_output_shapes");
  if (shapes_it != input_node->attr().end()) {
    const auto& shape_list = shapes_it->second.list();
    if (shape_list.shape_size() > 0) {
      const auto& input_shape = shape_list.shape(0);
      int input_rank = input_shape.dim_size();

      if (input_rank != 3) {
        VLOG(2) << "MusaTokenMixer: Input rank is " << input_rank
                << ", expected 3";
        return result;
      }

      int64 dim_T = input_shape.dim(1).size();
      int64 dim_D = input_shape.dim(2).size();

      // dim > 0 means the dimension is statically known
      if (dim_T > 0 && dim_T != num_T) {
        VLOG(2) << "MusaTokenMixer: Input dim[1]=" << dim_T
                << " != num_T=" << num_T;
        return result;
      }

      if (dim_D > 0 && dim_D != num_H * d_k) {
        VLOG(2) << "MusaTokenMixer: Input dim[2]=" << dim_D
                << " != num_H*d_k=" << num_H * d_k;
        return result;
      }
    }
  }

  // Also check Placeholder's "shape" attribute
  if (IsOp(*input_node, "Placeholder")) {
    auto shape_it = input_node->attr().find("shape");
    if (shape_it != input_node->attr().end()) {
      const auto& input_shape = shape_it->second.shape();
      int input_rank = input_shape.dim_size();

      if (input_rank > 0 && input_rank != 3) {
        VLOG(2) << "MusaTokenMixer: Placeholder rank is " << input_rank
                << ", expected 3";
        return result;
      }

      if (input_rank == 3) {
        int64 dim_D = input_shape.dim(2).size();
        if (dim_D > 0 && dim_D != num_H * d_k) {
          VLOG(2) << "MusaTokenMixer: Placeholder dim[2]=" << dim_D
                  << " != num_H*d_k=" << num_H * d_k;
          return result;
        }
      }
    }
  }
  // =========================================================================
  // Match successful!
  // =========================================================================
  result.matched = true;
  result.matched_nodes.push_back(reshape1_node);
  result.matched_nodes.push_back(transpose_node);
  result.matched_nodes.push_back(&reshape2_node);

  if (input_node) {
    result.captured_nodes["input"] = input_node;
  }

  result.captured_nodes["reshape1"] = reshape1_node;
  result.captured_nodes["transpose"] = transpose_node;
  result.captured_nodes["reshape2"] = &reshape2_node;
  result.captured_nodes["output"] = &reshape2_node;

  result.captured_attrs["num_T"] = std::to_string(num_T);
  result.captured_attrs["num_H"] = std::to_string(num_H);
  result.captured_attrs["d_k"] = std::to_string(d_k);

  VLOG(1) << "MusaTokenMixer: Matched pattern "
          << "input->reshape->transpose->reshape "
          << "(num_T=" << num_T << ", num_H=" << num_H << ", d_k=" << d_k
          << ")";

  return result;
}

Status MusaTokenMixerFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return ::tsl::errors::InvalidArgument("Invalid TokenMixer match result");
  }

  if (!IsKernelAvailable()) {
    return ::tsl::OkStatus();
  }

  // ---- Retrieve captured nodes ----
  auto output_it = match_result.captured_nodes.find("output");
  auto input_it = match_result.captured_nodes.find("input");
  auto reshape1_it = match_result.captured_nodes.find("reshape1");

  if (output_it == match_result.captured_nodes.end()) {
    return ::tsl::errors::InvalidArgument(
                  "Missing output node in TokenMixer pattern");
  }

  const NodeDef* output_node = output_it->second;

  // ---- Extract num_T / num_H / d_k ----
  auto num_T_it = match_result.captured_attrs.find("num_T");
  auto num_H_it = match_result.captured_attrs.find("num_H");
  auto d_k_it = match_result.captured_attrs.find("d_k");

  if (num_T_it == match_result.captured_attrs.end() ||
      num_H_it == match_result.captured_attrs.end() ||
      d_k_it == match_result.captured_attrs.end()) {
    return ::tsl::errors::InvalidArgument(
                  "Missing shape attributes in TokenMixer pattern");
  }

  int64 num_T = std::stoll(num_T_it->second);
  int64 num_H = std::stoll(num_H_it->second);
  int64 d_k = std::stoll(d_k_it->second);

  // ---- Duplicate-fusion guard ----
  std::string base_name = output_node->name();
  if (base_name.size() > 9 &&
      base_name.substr(base_name.size() - 9) == "_original") {
    base_name = base_name.substr(0, base_name.size() - 9);
  }

  for (const auto& node : graph->node()) {
    if (node.name() == base_name && node.op() == "MusaTokenMixer") {
      VLOG(1) << "MusaTokenMixer: " << base_name
              << " is already fused, skipping";
      return ::tsl::OkStatus();
    }
  }

  // ---- Create fused MusaTokenMixer node ----
  std::string fused_node_name = output_node->name() + "_fused_tokenmixer";
  VLOG(1) << "MusaTokenMixer: Creating fused node: " << fused_node_name;

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaTokenMixer");
  fused_node->set_device(output_node->device());

  // Input: directly reuse reshape1's data input (preserves port info)
  if (reshape1_it != match_result.captured_nodes.end() && reshape1_it->second &&
      reshape1_it->second->input_size() > 0) {
    fused_node->add_input(reshape1_it->second->input(0));
  } else {
    return ::tsl::errors::InvalidArgument("Cannot determine TokenMixer input");
  }

  // Attributes
  auto* attr = fused_node->mutable_attr();

  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    (*attr)["T"] = dtype_it->second;
  } else {
    (*attr)["T"].set_type(DT_FLOAT);
  }

  (*attr)["num_T"].set_i(num_T);
  (*attr)["num_H"].set_i(num_H);
  (*attr)["d_k"].set_i(d_k);

  // 1. rename origial output node
  std::string original_name = output_node->name();
  const_cast<NodeDef*>(output_node)->set_name(original_name + "_original");

  // 2. rename fused node to original name
  fused_node->set_name(original_name);
  VLOG(1) << "MusaTokenMixer: Fused node created as " << original_name
          << " (num_T=" << num_T << ", num_H=" << num_H << ", d_k=" << d_k
          << ")";
  std::set<std::string> nodes_to_remove;

  // 被重命名的 output（reshape2）节点
  nodes_to_remove.insert(original_name + "_original");

  // reshape1 和 transpose
  auto transpose_it = match_result.captured_nodes.find("transpose");
  if (transpose_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(transpose_it->second->name());
  }
  if (reshape1_it != match_result.captured_nodes.end()) {
    nodes_to_remove.insert(reshape1_it->second->name());
  }

  // 关联的 Const 节点（shape/perm）——仅在无其他消费者时删除
  // reshape1 的 shape const (input[1])
  if (reshape1_it != match_result.captured_nodes.end() &&
      reshape1_it->second->input_size() >= 2) {
    std::string shape1_name = GetProducerName(reshape1_it->second->input(1));
    if (CountConsumers(*graph, shape1_name) <= 1) {
      nodes_to_remove.insert(shape1_name);
    }
  }

  // transpose 的 perm const (input[1])
  if (transpose_it != match_result.captured_nodes.end() &&
      transpose_it->second->input_size() >= 2) {
    std::string perm_name = GetProducerName(transpose_it->second->input(1));
    if (CountConsumers(*graph, perm_name) <= 1) {
      nodes_to_remove.insert(perm_name);
    }
  }

  // reshape2 的 shape const (input[1])
  // 注意：原始 output_node 就是 reshape2
  if (output_node->input_size() >= 2) {
    std::string shape2_name = GetProducerName(output_node->input(1));
    if (CountConsumers(*graph, shape2_name) <= 1) {
      nodes_to_remove.insert(shape2_name);
    }
  }

  // 4. 从后往前删除节点（避免索引移位问题）
  // 先收集索引，然后按降序删除
  std::vector<int> indices_to_remove;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_remove.count(graph->node(i).name()) > 0) {
      indices_to_remove.push_back(i);
    }
  }
  std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());

  for (int idx : indices_to_remove) {
    VLOG(2) << "MusaTokenMixer: Removing node: " << graph->node(idx).name();
    FusionGraphUtils::RemoveNode(graph, idx);
  }

  return ::tsl::OkStatus();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MusaTokenMixerFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MusaTokenMixerFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow