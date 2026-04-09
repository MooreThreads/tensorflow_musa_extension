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

#include "mu/graph_fusion/matmul_bias_fusion.h"

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

std::string GetProducerName(const std::string& input) {
  return FusionGraphUtils::GetProducerNodeName(input);
}

const NodeDef* FindNode(const GraphDef& graph, const std::string& name) {
  return FusionGraphUtils::GetNodeByName(graph, name);
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string producer = GetProducerName(input);
  if (producer.empty()) {
    return nullptr;
  }
  return FindNode(graph, producer);
}

const NodeDef* ResolveIdentity(const GraphDef& graph, const NodeDef* node) {
  const NodeDef* cur = node;
  while (cur && IsOp(*cur, "Identity") && cur->input_size() > 0) {
    cur = FindProducer(graph, cur->input(0));
  }
  return cur;
}

int CountConsumers(const GraphDef& graph, const std::string& node_name) {
  int count = 0;
  for (const auto& node : graph.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      if (GetProducerName(node.input(i)) == node_name) {
        ++count;
      }
    }
  }
  return count;
}

bool IsConstRank1Vector(const NodeDef* node) {
  if (!node || !IsOp(*node, "Const")) {
    return false;
  }

  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end()) {
    return false;
  }

  const TensorProto& tensor = value_it->second.tensor();
  if (tensor.tensor_shape().dim_size() != 1) {
    return false;
  }
  return tensor.tensor_shape().dim(0).size() > 0;
}

}  // namespace

MatMulBiasFusion::MatMulBiasFusion() = default;

bool MatMulBiasFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MatMulBiasFusion::Match(const GraphDef& graph,
                                          int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& bias_add = graph.node(start_node_idx);
  if (!IsOp(bias_add, "BiasAdd") || bias_add.input_size() != 2) {
    return FusionMatchResult{};
  }

  const NodeDef* matmul_node = FindProducer(graph, bias_add.input(0));
  if (!matmul_node || !IsOp(*matmul_node, "MatMul") ||
      matmul_node->input_size() != 2) {
    return FusionMatchResult{};
  }

  // Keep fusion conservative to avoid duplicating MatMul work.
  if (CountConsumers(graph, matmul_node->name()) != 1) {
    return FusionMatchResult{};
  }

  const NodeDef* bias_node =
      ResolveIdentity(graph, FindProducer(graph, bias_add.input(1)));
  if (!IsConstRank1Vector(bias_node)) {
    return FusionMatchResult{};
  }

  auto t_it = bias_add.attr().find("T");
  if (t_it == bias_add.attr().end()) {
    return FusionMatchResult{};
  }

  bool transpose_a = false;
  bool transpose_b = false;
  auto ta_it = matmul_node->attr().find("transpose_a");
  if (ta_it != matmul_node->attr().end()) {
    transpose_a = ta_it->second.b();
  }
  auto tb_it = matmul_node->attr().find("transpose_b");
  if (tb_it != matmul_node->attr().end()) {
    transpose_b = tb_it->second.b();
  }

  FusionMatchResult result;
  result.matched = true;
  result.matched_nodes.push_back(&bias_add);
  result.matched_nodes.push_back(matmul_node);

  result.captured_nodes["bias_add"] = &bias_add;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["bias"] = bias_node;

  result.captured_attrs["input_a"] = matmul_node->input(0);
  result.captured_attrs["input_b"] = matmul_node->input(1);
  result.captured_attrs["bias_input"] = bias_add.input(1);
  result.captured_attrs["transpose_a"] = transpose_a ? "1" : "0";
  result.captured_attrs["transpose_b"] = transpose_b ? "1" : "0";

  return result;
}

Status MatMulBiasFusion::Apply(GraphDef* graph,
                               const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid MatMulBias match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto bias_add_it = match_result.captured_nodes.find("bias_add");
  auto matmul_it = match_result.captured_nodes.find("matmul");
  if (bias_add_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured nodes for MatMulBias fusion");
  }

  const std::string output_name = bias_add_it->second->name();
  const std::string matmul_name = matmul_it->second->name();

  const NodeDef* bias_add_node = FindNode(*graph, output_name);
  const NodeDef* matmul_node = FindNode(*graph, matmul_name);
  if (!bias_add_node || !matmul_node || !IsOp(*bias_add_node, "BiasAdd") ||
      !IsOp(*matmul_node, "MatMul") || bias_add_node->input_size() != 2 ||
      matmul_node->input_size() != 2) {
    return Status::OK();
  }

  // Re-validate single consumer for safety.
  if (CountConsumers(*graph, matmul_name) != 1) {
    return Status::OK();
  }

  const std::string output_device = bias_add_node->device();

  DataType dtype = DT_FLOAT;
  auto dtype_it = bias_add_node->attr().find("T");
  if (dtype_it != bias_add_node->attr().end()) {
    dtype = dtype_it->second.type();
  } else {
    auto matmul_dtype_it = matmul_node->attr().find("T");
    if (matmul_dtype_it != matmul_node->attr().end()) {
      dtype = matmul_dtype_it->second.type();
    }
  }

  bool transpose_a = false;
  bool transpose_b = false;
  auto ta_it = match_result.captured_attrs.find("transpose_a");
  if (ta_it != match_result.captured_attrs.end()) {
    transpose_a = (ta_it->second == "1");
  }
  auto tb_it = match_result.captured_attrs.find("transpose_b");
  if (tb_it != match_result.captured_attrs.end()) {
    transpose_b = (tb_it->second == "1");
  }

  std::string input_a = matmul_node->input(0);
  std::string input_b = matmul_node->input(1);
  std::string bias_input = bias_add_node->input(1);

  auto input_a_it = match_result.captured_attrs.find("input_a");
  if (input_a_it != match_result.captured_attrs.end() &&
      !input_a_it->second.empty()) {
    input_a = input_a_it->second;
  }
  auto input_b_it = match_result.captured_attrs.find("input_b");
  if (input_b_it != match_result.captured_attrs.end() &&
      !input_b_it->second.empty()) {
    input_b = input_b_it->second;
  }
  auto bias_input_it = match_result.captured_attrs.find("bias_input");
  if (bias_input_it != match_result.captured_attrs.end() &&
      !bias_input_it->second.empty()) {
    bias_input = bias_input_it->second;
  }

  const int bias_add_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (bias_add_idx < 0) {
    return Status::OK();
  }
  FusionGraphUtils::RemoveNode(graph, bias_add_idx);

  FusionGraphUtils::RemoveNodesIfUnused(graph, {matmul_name});

  NodeDef* fused = graph->add_node();
  fused->set_name(output_name);
  fused->set_op("MusaFusedMatMul");
  fused->set_device(output_device);
  fused->add_input(input_a);
  fused->add_input(input_b);
  fused->add_input(bias_input);

  auto* attr = fused->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["transpose_a"].set_b(transpose_a);
  (*attr)["transpose_b"].set_b(transpose_b);
  (*attr)["num_args"].set_i(0);

  auto* fused_ops = (*attr)["fused_ops"].mutable_list();
  fused_ops->add_s("BiasAdd");

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MatMulBiasFusion);
REGISTER_FUSION_KERNEL(MatMulBiasFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
