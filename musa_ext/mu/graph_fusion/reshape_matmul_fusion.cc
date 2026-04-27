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

#include "mu/graph_fusion/reshape_matmul_fusion.h"

#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

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

int GetOutputPort(const std::string& input) {
  if (input.empty()) return 0;

  std::string name = input;
  if (name[0] == '^') {
    name = name.substr(1);
  }
  size_t colon_pos = name.find(':');
  if (colon_pos == std::string::npos) {
    return 0;
  }
  return std::stoi(name.substr(colon_pos + 1));
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string node_name = GetCleanName(input);
  if (node_name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, node_name);
}

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

const NodeDef* GetConstLikeNode(const GraphDef& graph,
                                const std::string& input_name) {
  const NodeDef* node = FindProducer(graph, input_name);
  if (!node) return nullptr;

  while (node && node->op() == "Identity") {
    if (node->input_size() == 0) return nullptr;
    node = FindProducer(graph, node->input(0));
  }
  if (!node || node->op() != "Const") return nullptr;
  return node;
}

bool ExtractIntVector(const NodeDef* const_node, std::vector<int64_t>* values) {
  if (!const_node || !values || const_node->op() != "Const") {
    return false;
  }

  auto value_it = const_node->attr().find("value");
  if (value_it == const_node->attr().end()) {
    return false;
  }

  const TensorProto& tp = value_it->second.tensor();
  values->clear();

  if (tp.dtype() == DT_INT32) {
    if (tp.int_val_size() > 0) {
      for (int i = 0; i < tp.int_val_size(); ++i) {
        values->push_back(tp.int_val(i));
      }
      return true;
    }
    if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int32_t);
      const int32_t* data =
          reinterpret_cast<const int32_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        values->push_back(data[i]);
      }
      return true;
    }
  } else if (tp.dtype() == DT_INT64) {
    if (tp.int64_val_size() > 0) {
      for (int i = 0; i < tp.int64_val_size(); ++i) {
        values->push_back(tp.int64_val(i));
      }
      return true;
    }
    if (!tp.tensor_content().empty()) {
      const int n = tp.tensor_content().size() / sizeof(int64_t);
      const int64_t* data =
          reinterpret_cast<const int64_t*>(tp.tensor_content().data());
      for (int i = 0; i < n; ++i) {
        values->push_back(data[i]);
      }
      return true;
    }
  }

  if (values->empty() && tp.tensor_shape().dim_size() == 0) {
    if (tp.dtype() == DT_INT32 && tp.int_val_size() == 1) {
      values->push_back(tp.int_val(0));
      return true;
    }
    if (tp.dtype() == DT_INT64 && tp.int64_val_size() == 1) {
      values->push_back(tp.int64_val(0));
      return true;
    }
  }

  return false;
}

bool ExtractScalarInt(const NodeDef* const_node, int64_t* value) {
  std::vector<int64_t> values;
  if (!ExtractIntVector(const_node, &values) || values.size() != 1) {
    return false;
  }
  *value = values[0];
  return true;
}

bool ExtractConst2DShape(const NodeDef* const_node, int64_t* dim0,
                         int64_t* dim1) {
  if (!const_node || !dim0 || !dim1 || const_node->op() != "Const") {
    return false;
  }

  auto value_it = const_node->attr().find("value");
  if (value_it == const_node->attr().end()) {
    return false;
  }
  const TensorProto& tp = value_it->second.tensor();
  const TensorShapeProto& shape = tp.tensor_shape();
  if (shape.dim_size() != 2) {
    return false;
  }
  *dim0 = shape.dim(0).size();
  *dim1 = shape.dim(1).size();
  return true;
}

bool ExtractOutputShapeDims(const NodeDef* node, int output_port,
                           std::vector<int64_t>* dims) {
  if (!node || !dims) {
    return false;
  }

  dims->clear();

  auto out_shapes_it = node->attr().find("_output_shapes");
  if (out_shapes_it != node->attr().end()) {
    const auto& list = out_shapes_it->second.list().shape();
    if (output_port >= 0 && output_port < list.size()) {
      const auto& shape = list.Get(output_port);
      if (!shape.unknown_rank()) {
        for (const auto& dim : shape.dim()) {
          dims->push_back(dim.size());
        }
        return true;
      }
    }
  }

  if (output_port == 0) {
    auto shape_it = node->attr().find("shape");
    if (shape_it != node->attr().end()) {
      const auto& shape = shape_it->second.shape();
      if (!shape.unknown_rank()) {
        for (const auto& dim : shape.dim()) {
          dims->push_back(dim.size());
        }
        return true;
      }
    }
  }

  return false;
}

bool MatchFlattenShape(const NodeDef* shape_const, int64_t expected_k) {
  std::vector<int64_t> dims;
  if (!ExtractIntVector(shape_const, &dims) || dims.size() != 2) {
    return false;
  }
  return dims[0] == -1 && dims[1] == expected_k;
}

bool MatchRestoreShapePack(const GraphDef& graph, const std::string& source_input,
                           const NodeDef* pack_node, int64_t expected_n,
                           std::vector<std::string>* removable_node_names) {
  const NodeDef* source_node = FindProducer(graph, source_input);
  if (!source_node || !pack_node || !removable_node_names ||
      pack_node->op() != "Pack" || pack_node->input_size() < 2) {
    return false;
  }

  const int source_port = GetOutputPort(source_input);
  std::vector<int64_t> source_shape_dims;
  const bool has_source_shape =
      ExtractOutputShapeDims(source_node, source_port, &source_shape_dims);
  int inferred_rank_from_unpack = -1;

  const NodeDef* trailing_dim = GetConstLikeNode(
      graph, pack_node->input(pack_node->input_size() - 1));
  int64_t output_dim = 0;
  if (!ExtractScalarInt(trailing_dim, &output_dim) || output_dim != expected_n) {
    return false;
  }
  removable_node_names->push_back(trailing_dim->name());

  for (int i = 0; i < pack_node->input_size() - 1; ++i) {
    const NodeDef* input_node = FindProducer(graph, pack_node->input(i));
    if (!input_node) {
      return false;
    }

    if (input_node->op() == "Unpack") {
      if (GetOutputPort(pack_node->input(i)) != i) {
        return false;
      }
      if (input_node->input_size() != 1) {
        return false;
      }
      auto num_it = input_node->attr().find("num");
      if (num_it == input_node->attr().end() || num_it->second.i() <= i) {
        return false;
      }
      inferred_rank_from_unpack = static_cast<int>(num_it->second.i());

      const NodeDef* shape_of_source = FindProducer(graph, input_node->input(0));
      if (!shape_of_source || shape_of_source->op() != "Shape" ||
          shape_of_source->input_size() != 1 ||
          GetCleanName(shape_of_source->input(0)) != GetCleanName(source_input) ||
          GetOutputPort(shape_of_source->input(0)) != source_port) {
        return false;
      }

      removable_node_names->push_back(input_node->name());
      removable_node_names->push_back(shape_of_source->name());
      continue;
    }

    if (input_node->op() == "Const") {
      int64_t dim_value = 0;
      if (!ExtractScalarInt(input_node, &dim_value)) {
        return false;
      }
      if (has_source_shape && i < source_shape_dims.size() &&
          source_shape_dims[i] >= 0) {
        if (source_shape_dims[i] != dim_value) {
          return false;
        }
      } else {
        // Common production pattern: rank-3 source tensor where only the first
        // prefix dim stays dynamic and the second prefix dim has already been
        // folded into a Const in restore_shape Pack. In this case runtime
        // correctness still requires the Const to equal source.shape[1], and
        // using source.shape() inside the fused op reproduces the same result.
        const bool allow_rank3_tail_const =
            pack_node->input_size() == 3 && i == 1 && inferred_rank_from_unpack == 3;
        if (!allow_rank3_tail_const) {
          return false;
        }
      }
      removable_node_names->push_back(input_node->name());
      continue;
    }

    return false;
  }

  return true;
}

}  // namespace

bool MusaReshapeMatMulFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaReshapeMatMulFusion::Match(const GraphDef& graph,
                                                 int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& reshape_2 = graph.node(start_node_idx);
  if (!IsOp(reshape_2, "Reshape") || HasOriginalSuffix(reshape_2.name()) ||
      reshape_2.input_size() != 2) {
    return result;
  }

  const NodeDef* matmul_node = FindProducer(graph, reshape_2.input(0));
  const NodeDef* pack_node = FindProducer(graph, reshape_2.input(1));
  if (!matmul_node || !IsOp(*matmul_node, "MatMul") || !pack_node ||
      !IsOp(*pack_node, "Pack")) {
    return result;
  }

  if (matmul_node->input_size() != 2) {
    return result;
  }

  bool transpose_a = false;
  bool transpose_b = false;
  auto trans_a_it = matmul_node->attr().find("transpose_a");
  if (trans_a_it != matmul_node->attr().end()) {
    transpose_a = trans_a_it->second.b();
  }
  auto trans_b_it = matmul_node->attr().find("transpose_b");
  if (trans_b_it != matmul_node->attr().end()) {
    transpose_b = trans_b_it->second.b();
  }
  if (transpose_a) {
    return result;
  }

  const NodeDef* reshape_1 = FindProducer(graph, matmul_node->input(0));
  const NodeDef* weight_node = GetConstLikeNode(graph, matmul_node->input(1));
  if (!reshape_1 || !IsOp(*reshape_1, "Reshape") || !weight_node ||
      reshape_1->input_size() != 2) {
    return result;
  }

  const NodeDef* source_node = FindProducer(graph, reshape_1->input(0));
  const NodeDef* reshape_1_shape = GetConstLikeNode(graph, reshape_1->input(1));
  if (!source_node || !reshape_1_shape) {
    return result;
  }

  int64_t weight_dim0 = 0;
  int64_t weight_dim1 = 0;
  if (!ExtractConst2DShape(weight_node, &weight_dim0, &weight_dim1)) {
    return result;
  }

  const int64_t k = transpose_b ? weight_dim1 : weight_dim0;
  const int64_t n = transpose_b ? weight_dim0 : weight_dim1;
  if (!MatchFlattenShape(reshape_1_shape, k)) {
    return result;
  }

  std::vector<std::string> removable_shape_nodes;
  if (!MatchRestoreShapePack(graph, reshape_1->input(0), pack_node, n,
                             &removable_shape_nodes)) {
    return result;
  }

  if (FusionGraphUtils::HasAnyConsumer(graph, reshape_1->name())) {
    // Count consumers precisely so shared intermediates are not fused away.
    int consumers = 0;
    for (const auto& node : graph.node()) {
      for (const auto& input : node.input()) {
        if (GetCleanName(input) == reshape_1->name()) {
          ++consumers;
        }
      }
    }
    if (consumers != 1) {
      return result;
    }
  }

  int matmul_consumers = 0;
  for (const auto& node : graph.node()) {
    for (const auto& input : node.input()) {
      if (GetCleanName(input) == matmul_node->name()) {
        ++matmul_consumers;
      }
    }
  }
  if (matmul_consumers != 1) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&reshape_2, matmul_node, reshape_1, pack_node};
  result.captured_nodes["output"] = &reshape_2;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["reshape_1"] = reshape_1;
  result.captured_nodes["shape_pack"] = pack_node;
  result.captured_nodes["weight"] = weight_node;
  result.captured_nodes["reshape_1_shape"] = reshape_1_shape;
  result.captured_nodes["source"] = source_node;

  result.captured_attrs["source_input"] = reshape_1->input(0);
  result.captured_attrs["weight_input"] = matmul_node->input(1);
  result.captured_attrs["transpose_b"] = transpose_b ? "1" : "0";
  result.captured_attrs["num_shape_nodes"] =
      std::to_string(removable_shape_nodes.size());
  for (size_t i = 0; i < removable_shape_nodes.size(); ++i) {
    result.captured_attrs["shape_node_" + std::to_string(i)] =
        removable_shape_nodes[i];
  }
  return result;
}

Status MusaReshapeMatMulFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid MusaReshapeMatMul match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto source_it = match_result.captured_attrs.find("source_input");
  auto weight_it = match_result.captured_attrs.find("weight_input");
  auto transpose_b_it = match_result.captured_attrs.find("transpose_b");
  auto shape_nodes_it = match_result.captured_attrs.find("num_shape_nodes");
  if (output_it == match_result.captured_nodes.end() ||
      source_it == match_result.captured_attrs.end() ||
      weight_it == match_result.captured_attrs.end() ||
      transpose_b_it == match_result.captured_attrs.end() ||
      shape_nodes_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required captured fields for MusaReshapeMatMul");
  }

  const NodeDef* output_node = output_it->second;
  const std::string output_name = output_node->name();

  for (const auto& node : graph->node()) {
    if (node.name() == output_name && node.op() == "MusaReshapeMatMul") {
      return Status::OK();
    }
  }

  const int output_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (output_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate output node for MusaReshapeMatMul fusion");
  }

  NodeDef* original_output_node = graph->mutable_node(output_idx);
  const std::string output_device = original_output_node->device();
  DataType dtype = DT_FLOAT;
  auto t_it = original_output_node->attr().find("T");
  if (t_it != original_output_node->attr().end()) {
    dtype = t_it->second.type();
  }

  const std::string original_output_name = output_name + "_original";
  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(output_name);
  fused_node->set_op("MusaReshapeMatMul");
  fused_node->set_device(output_device);
  fused_node->add_input(source_it->second);
  fused_node->add_input(weight_it->second);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["transpose_b"].set_b(transpose_b_it->second == "1");

  std::unordered_set<std::string> removable_names = {
      original_output_name,
      match_result.captured_nodes.at("matmul")->name(),
      match_result.captured_nodes.at("reshape_1")->name(),
      match_result.captured_nodes.at("shape_pack")->name(),
      match_result.captured_nodes.at("reshape_1_shape")->name(),
  };

  int num_shape_nodes = 0;
  try {
    num_shape_nodes = std::stoi(shape_nodes_it->second);
  } catch (...) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid num_shape_nodes for MusaReshapeMatMul");
  }
  for (int i = 0; i < num_shape_nodes; ++i) {
    auto it = match_result.captured_attrs.find("shape_node_" + std::to_string(i));
    if (it != match_result.captured_attrs.end()) {
      removable_names.insert(it->second);
    }
  }

  std::vector<std::string> removable_list(removable_names.begin(),
                                          removable_names.end());
  const int removed = FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_list, {output_name, source_it->second, weight_it->second});

  VLOG(1) << "MusaReshapeMatMulFusion: fused output=" << output_name
          << ", removed=" << removed;

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaReshapeMatMulFusion);
REGISTER_FUSION_KERNEL(MusaReshapeMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
