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

#include "kernels/fusion/pln_cascade_fusion.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "kernels/fusion/fusion_pattern_manager.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

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

NodeDef* FindMutableNode(GraphDef* graph, const std::string& name) {
  if (!graph) {
    return nullptr;
  }
  const int node_idx = FusionGraphUtils::FindNodeIndex(*graph, name);
  if (node_idx < 0) {
    return nullptr;
  }
  return graph->mutable_node(node_idx);
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

bool ExtractPlnGroup(const std::string& node_name, std::string* group) {
  const size_t pln_pos = node_name.find("/pln");
  if (pln_pos == std::string::npos) {
    return false;
  }
  const size_t slash_after_group = node_name.find('/', pln_pos + 1);
  if (slash_after_group == std::string::npos) {
    return false;
  }

  const std::string token =
      node_name.substr(pln_pos + 1, slash_after_group - (pln_pos + 1));
  if (token.rfind("pln1_", 0) != 0 && token.rfind("pln2_", 0) != 0) {
    return false;
  }

  *group = node_name.substr(0, slash_after_group);
  return true;
}

bool BelongsToGroup(const std::string& node_name, const std::string& group) {
  const std::string prefix = group + "/";
  return node_name.rfind(prefix, 0) == 0;
}

std::string MakePairKey(const std::string& add_name,
                        const std::string& bias_name) {
  return add_name + "|" + bias_name;
}

bool ReadFloatTensor(const TensorProto& tensor, std::vector<int64>* dims,
                     std::vector<float>* values) {
  if (tensor.dtype() != DT_FLOAT) {
    return false;
  }

  dims->clear();
  int64 numel = 1;
  if (tensor.tensor_shape().dim_size() > 0) {
    for (const auto& d : tensor.tensor_shape().dim()) {
      dims->push_back(d.size());
      numel *= d.size();
    }
  }

  if (!tensor.tensor_content().empty()) {
    const size_t expect_size = static_cast<size_t>(numel) * sizeof(float);
    if (tensor.tensor_content().size() != expect_size) {
      return false;
    }
    values->resize(numel);
    std::memcpy(values->data(), tensor.tensor_content().data(), expect_size);
    return true;
  }

  if (tensor.float_val_size() <= 0) {
    return false;
  }
  if (tensor.float_val_size() == 1 && numel > 1) {
    values->assign(numel, tensor.float_val(0));
    return true;
  }
  if (tensor.float_val_size() != numel) {
    return false;
  }

  values->resize(numel);
  for (int i = 0; i < tensor.float_val_size(); ++i) {
    (*values)[i] = tensor.float_val(i);
  }
  return true;
}

bool ExtractConstVector(const NodeDef* node, std::vector<float>* out) {
  if (!node || !IsOp(*node, "Const")) {
    return false;
  }
  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end()) {
    return false;
  }

  std::vector<int64> dims;
  std::vector<float> values;
  if (!ReadFloatTensor(value_it->second.tensor(), &dims, &values)) {
    return false;
  }

  if (dims.empty()) {
    out->assign(1, values[0]);
    return true;
  }
  if (dims.size() == 1) {
    *out = std::move(values);
    return true;
  }
  if (dims.size() == 2 && dims[0] == 1) {
    out->assign(values.begin(), values.begin() + dims[1]);
    return true;
  }
  return false;
}

bool ExtractConstIntScalar(const NodeDef* node, int64* out) {
  if (!node || !IsOp(*node, "Const")) {
    return false;
  }
  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end()) {
    return false;
  }
  const TensorProto& tensor = value_it->second.tensor();
  if (tensor.dtype() == DT_INT32) {
    if (tensor.int_val_size() > 0) {
      *out = tensor.int_val(0);
      return true;
    }
    if (tensor.tensor_content().size() >= sizeof(int32_t)) {
      int32_t v;
      std::memcpy(&v, tensor.tensor_content().data(), sizeof(int32_t));
      *out = static_cast<int64>(v);
      return true;
    }
    return false;
  }

  if (tensor.dtype() == DT_INT64) {
    if (tensor.int64_val_size() > 0) {
      *out = tensor.int64_val(0);
      return true;
    }
    if (tensor.tensor_content().size() >= sizeof(int64)) {
      int64 v;
      std::memcpy(&v, tensor.tensor_content().data(), sizeof(int64));
      *out = v;
      return true;
    }
    return false;
  }
  return false;
}

bool ExtractBiasVector(const GraphDef& graph, const NodeDef* bias_node,
                       std::vector<float>* out) {
  const NodeDef* resolved_bias = ResolveIdentity(graph, bias_node);
  if (!resolved_bias) {
    return false;
  }

  if (IsOp(*resolved_bias, "Const")) {
    return ExtractConstVector(resolved_bias, out);
  }

  if (!IsOp(*resolved_bias, "StridedSlice") ||
      resolved_bias->input_size() < 4) {
    return false;
  }

  const NodeDef* data_node =
      ResolveIdentity(graph, FindProducer(graph, resolved_bias->input(0)));
  const NodeDef* begin_node =
      ResolveIdentity(graph, FindProducer(graph, resolved_bias->input(1)));
  const NodeDef* end_node =
      ResolveIdentity(graph, FindProducer(graph, resolved_bias->input(2)));
  const NodeDef* stride_node =
      ResolveIdentity(graph, FindProducer(graph, resolved_bias->input(3)));
  if (!data_node || !begin_node || !end_node || !stride_node) {
    return false;
  }

  int64 begin = 0;
  int64 end = 0;
  int64 stride = 0;
  if (!ExtractConstIntScalar(begin_node, &begin) ||
      !ExtractConstIntScalar(end_node, &end) ||
      !ExtractConstIntScalar(stride_node, &stride)) {
    return false;
  }
  if (stride != 1 || end <= begin) {
    return false;
  }

  if (!IsOp(*data_node, "Const")) {
    return false;
  }
  auto value_it = data_node->attr().find("value");
  if (value_it == data_node->attr().end()) {
    return false;
  }

  std::vector<int64> dims;
  std::vector<float> values;
  if (!ReadFloatTensor(value_it->second.tensor(), &dims, &values)) {
    return false;
  }

  if (dims.size() == 1) {
    if (begin < 0 || end > dims[0] || end - begin <= 0) {
      return false;
    }
    out->assign(values.begin() + begin, values.begin() + end);
    return true;
  }

  if (dims.size() != 2) {
    return false;
  }
  const int64 rows = dims[0];
  const int64 cols = dims[1];
  if (begin < 0 || begin >= rows || end != begin + 1) {
    return false;
  }

  const size_t offset = static_cast<size_t>(begin * cols);
  out->assign(values.begin() + offset, values.begin() + offset + cols);
  return true;
}

void AddFloatTableConstNode(GraphDef* graph, const std::string& node_name,
                            const std::string& device,
                            const std::vector<float>& table_values, int64 rows,
                            int64 cols) {
  NodeDef* node = graph->add_node();
  node->set_name(node_name);
  node->set_op("Const");
  node->set_device(device);

  auto* attr = node->mutable_attr();
  (*attr)["dtype"].set_type(DT_FLOAT);
  TensorProto* tensor = (*attr)["value"].mutable_tensor();
  tensor->set_dtype(DT_FLOAT);
  tensor->mutable_tensor_shape()->add_dim()->set_size(rows);
  tensor->mutable_tensor_shape()->add_dim()->set_size(cols);
  tensor->set_tensor_content(
      std::string(reinterpret_cast<const char*>(table_values.data()),
                  table_values.size() * sizeof(float)));
}

void UpdateFloatTableConstNode(NodeDef* node,
                               const std::vector<float>& table_values,
                               int64 rows, int64 cols) {
  auto* attr = node->mutable_attr();
  (*attr)["dtype"].set_type(DT_FLOAT);

  TensorProto* tensor = (*attr)["value"].mutable_tensor();
  tensor->Clear();
  tensor->set_dtype(DT_FLOAT);
  tensor->mutable_tensor_shape()->add_dim()->set_size(rows);
  tensor->mutable_tensor_shape()->add_dim()->set_size(cols);
  tensor->set_tensor_content(
      std::string(reinterpret_cast<const char*>(table_values.data()),
                  table_values.size() * sizeof(float)));
}

bool TryExtractPairFromMapNode(const GraphDef& graph, const NodeDef& map_node,
                               std::string* add_name, std::string* bias_name,
                               std::vector<float>* add_vec,
                               std::vector<float>* bias_vec) {
  if (!IsOp(map_node, "MusaShiftedAffineMap") || map_node.input_size() != 3) {
    return false;
  }

  *add_name = GetProducerName(map_node.input(0));
  *bias_name = GetProducerName(map_node.input(2));
  if (add_name->empty() || bias_name->empty()) {
    return false;
  }

  const NodeDef* add_node = ResolveIdentity(graph, FindNode(graph, *add_name));
  const NodeDef* bias_node = FindNode(graph, *bias_name);

  if (!ExtractConstVector(add_node, add_vec) ||
      !ExtractBiasVector(graph, bias_node, bias_vec)) {
    return false;
  }

  return !add_vec->empty() && add_vec->size() == bias_vec->size();
}

bool TryExtractPairFromPendingShiftedAffine(
    const GraphDef& graph, const NodeDef& output_add_node,
    std::string* add_name, std::string* bias_name, std::vector<float>* add_vec,
    std::vector<float>* bias_vec, std::string* data_input_edge = nullptr,
    std::string* mul_name = nullptr, std::string* add_input_edge_out = nullptr,
    std::string* bias_input_edge_out = nullptr) {
  if (!IsOp(output_add_node, "AddV2") || output_add_node.input_size() != 2) {
    return false;
  }

  const NodeDef* add_in0 = FindProducer(graph, output_add_node.input(0));
  const NodeDef* add_in1 = FindProducer(graph, output_add_node.input(1));
  if (!add_in0 || !add_in1) {
    return false;
  }

  const NodeDef* mul_node = nullptr;
  std::string bias_input_edge;
  if (IsOp(*add_in0, "Mul")) {
    mul_node = add_in0;
    bias_input_edge = output_add_node.input(1);
  } else if (IsOp(*add_in1, "Mul")) {
    mul_node = add_in1;
    bias_input_edge = output_add_node.input(0);
  } else {
    return false;
  }

  if (!mul_node || mul_node->input_size() != 2) {
    return false;
  }
  if (mul_name != nullptr) {
    *mul_name = mul_node->name();
  }

  const NodeDef* mul_in0 = FindProducer(graph, mul_node->input(0));
  const NodeDef* mul_in1 = FindProducer(graph, mul_node->input(1));
  if (!mul_in0 || !mul_in1) {
    return false;
  }

  std::string add_input_edge;
  std::string passthrough_input_edge;
  const NodeDef* resolved_mul_in0 = ResolveIdentity(graph, mul_in0);
  const NodeDef* resolved_mul_in1 = ResolveIdentity(graph, mul_in1);
  if (resolved_mul_in0 && IsOp(*resolved_mul_in0, "Const") &&
      resolved_mul_in1 && !IsOp(*resolved_mul_in1, "Const")) {
    add_input_edge = mul_node->input(0);
    passthrough_input_edge = mul_node->input(1);
  } else if (resolved_mul_in1 && IsOp(*resolved_mul_in1, "Const") &&
             resolved_mul_in0 && !IsOp(*resolved_mul_in0, "Const")) {
    add_input_edge = mul_node->input(1);
    passthrough_input_edge = mul_node->input(0);
  } else {
    return false;
  }

  if (data_input_edge != nullptr) {
    *data_input_edge = passthrough_input_edge;
  }
  if (add_input_edge_out != nullptr) {
    *add_input_edge_out = add_input_edge;
  }
  if (bias_input_edge_out != nullptr) {
    *bias_input_edge_out = bias_input_edge;
  }

  *add_name = GetProducerName(add_input_edge);
  *bias_name = GetProducerName(bias_input_edge);
  if (add_name->empty() || bias_name->empty()) {
    return false;
  }

  const NodeDef* add_node = ResolveIdentity(graph, FindNode(graph, *add_name));
  const NodeDef* bias_node = FindNode(graph, *bias_name);
  if (!ExtractConstVector(add_node, add_vec) ||
      !ExtractBiasVector(graph, bias_node, bias_vec)) {
    return false;
  }

  return !add_vec->empty() && add_vec->size() == bias_vec->size();
}

void CollectBiasCleanupCandidates(const GraphDef& graph,
                                  const std::string& bias_name,
                                  std::vector<std::string>* removable) {
  const NodeDef* bias_node = FindNode(graph, bias_name);
  if (!bias_node || !IsOp(*bias_node, "StridedSlice")) {
    return;
  }

  removable->push_back(bias_name);
  for (int i = 0; i < bias_node->input_size(); ++i) {
    const std::string producer_name = GetProducerName(bias_node->input(i));
    if (producer_name.empty()) {
      continue;
    }
    removable->push_back(producer_name);
    const NodeDef* producer = FindNode(graph, producer_name);
    if (producer && IsOp(*producer, "Identity") && producer->input_size() > 0) {
      const std::string origin_name = GetProducerName(producer->input(0));
      if (!origin_name.empty()) {
        removable->push_back(origin_name);
      }
    }
  }
}

}  // namespace

MusaPlnCascadeFusion::MusaPlnCascadeFusion() = default;

bool MusaPlnCascadeFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaPlnCascadeFusion::Match(const GraphDef& graph,
                                              int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& select = graph.node(start_node_idx);
  if (!IsOp(select, "Select") || select.input_size() != 3) {
    return FusionMatchResult{};
  }

  std::string group;
  if (!ExtractPlnGroup(select.name(), &group)) {
    return FusionMatchResult{};
  }

  const NodeDef* cond_node = FindProducer(graph, select.input(0));
  if (!cond_node ||
      (!IsOp(*cond_node, "Equal") && !IsOp(*cond_node, "LogicalOr"))) {
    return FusionMatchResult{};
  }

  const NodeDef* true_node = FindProducer(graph, select.input(1));
  const NodeDef* false_node = FindProducer(graph, select.input(2));
  if (!true_node || !false_node) {
    return FusionMatchResult{};
  }

  const NodeDef* affine_node = nullptr;
  const NodeDef* passthrough_node = nullptr;
  bool select_on_true = true;
  std::string norm_input;
  bool is_pending_affine = false;
  std::string add_name;
  std::string bias_name;
  std::string add_input_edge;
  std::string bias_input_edge;
  std::string pending_mul_name;

  auto try_match_pending =
      [&](const NodeDef* pending_add_node, const NodeDef* passthrough_candidate,
          bool selected_on_true, const std::string& norm_edge) -> bool {
    if (!pending_add_node || !passthrough_candidate ||
        !IsOp(*pending_add_node, "AddV2")) {
      return false;
    }

    std::vector<float> add_vec;
    std::vector<float> bias_vec;
    std::string data_edge;
    std::string local_mul_name;
    std::string local_add_edge;
    std::string local_bias_edge;
    std::string local_add_name;
    std::string local_bias_name;
    if (!TryExtractPairFromPendingShiftedAffine(
            graph, *pending_add_node, &local_add_name, &local_bias_name,
            &add_vec, &bias_vec, &data_edge, &local_mul_name, &local_add_edge,
            &local_bias_edge)) {
      return false;
    }

    if (GetProducerName(data_edge) != passthrough_candidate->name()) {
      return false;
    }
    if (!BelongsToGroup(pending_add_node->name(), group) ||
        CountConsumers(graph, pending_add_node->name()) != 1) {
      return false;
    }

    affine_node = pending_add_node;
    passthrough_node = passthrough_candidate;
    select_on_true = selected_on_true;
    norm_input = norm_edge;
    is_pending_affine = true;
    add_name = local_add_name;
    bias_name = local_bias_name;
    add_input_edge = local_add_edge;
    bias_input_edge = local_bias_edge;
    pending_mul_name = local_mul_name;
    return true;
  };

  if (IsOp(*true_node, "MusaShiftedAffineMap")) {
    affine_node = true_node;
    passthrough_node = false_node;
    select_on_true = true;
    norm_input = select.input(2);
  } else if (IsOp(*false_node, "MusaShiftedAffineMap")) {
    affine_node = false_node;
    passthrough_node = true_node;
    select_on_true = false;
    norm_input = select.input(1);
  } else if (try_match_pending(true_node, false_node, true, select.input(2))) {
    // matched pending AddV2+Mul affine on true branch
  } else if (try_match_pending(false_node, true_node, false, select.input(1))) {
    // matched pending AddV2+Mul affine on false branch
  } else {
    return FusionMatchResult{};
  }

  if (!is_pending_affine) {
    if (!affine_node || affine_node->input_size() != 3) {
      return FusionMatchResult{};
    }
    if (GetProducerName(affine_node->input(1)) != passthrough_node->name()) {
      return FusionMatchResult{};
    }
    if (!BelongsToGroup(affine_node->name(), group)) {
      return FusionMatchResult{};
    }
    if (CountConsumers(graph, affine_node->name()) != 1) {
      return FusionMatchResult{};
    }

    add_name = GetProducerName(affine_node->input(0));
    bias_name = GetProducerName(affine_node->input(2));
    if (add_name.empty() || bias_name.empty()) {
      return FusionMatchResult{};
    }
    add_input_edge = affine_node->input(0);
    bias_input_edge = affine_node->input(2);
  }

  auto t_it = select.attr().find("T");
  if (t_it == select.attr().end() || t_it->second.type() != DT_FLOAT) {
    return FusionMatchResult{};
  }

  if (add_name.empty() || bias_name.empty()) {
    return FusionMatchResult{};
  }

  FusionMatchResult result;
  result.matched = true;
  result.matched_nodes.push_back(&select);
  result.matched_nodes.push_back(affine_node);

  result.captured_nodes["select"] = &select;
  result.captured_nodes["affine"] = affine_node;
  result.captured_nodes["condition"] = cond_node;
  result.captured_nodes["passthrough"] = passthrough_node;

  result.captured_attrs["group"] = group;
  result.captured_attrs["select_on_true"] = select_on_true ? "1" : "0";
  result.captured_attrs["norm_input"] = norm_input;
  result.captured_attrs["gate_input"] = select.input(0);
  result.captured_attrs["add_name"] = add_name;
  result.captured_attrs["bias_name"] = bias_name;
  result.captured_attrs["affine_name"] = affine_node->name();
  result.captured_attrs["is_pending_affine"] = is_pending_affine ? "1" : "0";
  result.captured_attrs["add_input_edge"] = add_input_edge;
  result.captured_attrs["bias_input_edge"] = bias_input_edge;
  if (!pending_mul_name.empty()) {
    result.captured_attrs["pending_mul_name"] = pending_mul_name;
  }

  return result;
}

bool MusaPlnCascadeFusion::BuildGroupTables(GraphDef* graph,
                                            const std::string& group,
                                            GroupTableInfo* info) const {
  if (!graph || !info) {
    return false;
  }

  info->initialized = true;
  info->enabled = false;

  // If cached table nodes are missing in current graph, rebuild this group's
  // cache from scratch.
  if ((!info->add_table_node.empty() &&
       !FindNode(*graph, info->add_table_node)) ||
      (!info->bias_table_node.empty() &&
       !FindNode(*graph, info->bias_table_node))) {
    *info = GroupTableInfo{};
    info->initialized = true;
  }

  struct CandidateRow {
    std::string key;
    std::vector<float> add_vec;
    std::vector<float> bias_vec;
  };

  std::vector<CandidateRow> candidate_rows;
  std::unordered_set<std::string> seen_keys;
  std::string table_device;

  auto try_add_candidate = [&](const std::string& key,
                               const std::vector<float>& add_vec,
                               const std::vector<float>& bias_vec,
                               const std::string& device) {
    if (key.empty() || add_vec.empty() || add_vec.size() != bias_vec.size()) {
      return;
    }
    if (!seen_keys.insert(key).second) {
      return;
    }
    candidate_rows.push_back(CandidateRow{key, add_vec, bias_vec});
    if (table_device.empty() && !device.empty()) {
      table_device = device;
    }
  };

  // Collect rows from already-fused MusaShiftedAffineMap nodes.
  for (const auto& node : graph->node()) {
    if (node.op() == "MusaShiftedAffineMap" &&
        BelongsToGroup(node.name(), group)) {
      std::string add_name;
      std::string bias_name;
      std::vector<float> add_vec;
      std::vector<float> bias_vec;
      if (TryExtractPairFromMapNode(*graph, node, &add_name, &bias_name,
                                    &add_vec, &bias_vec)) {
        try_add_candidate(MakePairKey(add_name, bias_name), add_vec, bias_vec,
                          node.device());
      }
    }
  }

  // Also collect rows from still-unfused ShiftedAffine candidates (AddV2
  // pattern) so table growth does not depend on exact pass interleaving.
  for (const auto& node : graph->node()) {
    if (node.op() == "AddV2" && BelongsToGroup(node.name(), group)) {
      std::string add_name;
      std::string bias_name;
      std::vector<float> add_vec;
      std::vector<float> bias_vec;
      if (TryExtractPairFromPendingShiftedAffine(
              *graph, node, &add_name, &bias_name, &add_vec, &bias_vec)) {
        try_add_candidate(MakePairKey(add_name, bias_name), add_vec, bias_vec,
                          node.device());
      }
    }
  }

  int64_t row_width = info->row_width;
  bool appended_new_rows = false;
  for (const auto& row : candidate_rows) {
    const int64_t current_width = static_cast<int64_t>(row.add_vec.size());
    if (row_width < 0) {
      row_width = current_width;
    }
    if (row_width != current_width) {
      VLOG(1) << "MusaPlnCascadeFusion: skip row with width " << current_width
              << " in group '" << group << "' (expected width=" << row_width
              << ")";
      continue;
    }

    if (info->pair_to_index.find(row.key) != info->pair_to_index.end()) {
      continue;
    }

    const int index = static_cast<int>(info->pair_to_index.size());
    info->pair_to_index[row.key] = index;
    info->add_table_values.insert(info->add_table_values.end(),
                                  row.add_vec.begin(), row.add_vec.end());
    info->bias_table_values.insert(info->bias_table_values.end(),
                                   row.bias_vec.begin(), row.bias_vec.end());
    appended_new_rows = true;
  }
  info->row_width = row_width;

  if (info->row_width <= 0 || info->pair_to_index.empty()) {
    return false;
  }

  const int64 rows = static_cast<int64>(info->pair_to_index.size());
  const size_t expected_values_size =
      static_cast<size_t>(rows * info->row_width);
  if (info->add_table_values.size() != expected_values_size ||
      info->bias_table_values.size() != expected_values_size) {
    LOG(WARNING)
        << "MusaPlnCascadeFusion: inconsistent cached table for group '"
        << group << "' (rows=" << rows << ", row_width=" << info->row_width
        << ", add_values=" << info->add_table_values.size()
        << ", bias_values=" << info->bias_table_values.size() << ")";
    info->enabled = false;
    return false;
  }

  if (info->add_table_node.empty()) {
    info->add_table_node = group + "/pln_cascade_add_table";
  }
  if (info->bias_table_node.empty()) {
    info->bias_table_node = group + "/pln_cascade_bias_table";
  }

  NodeDef* add_table_node = FindMutableNode(graph, info->add_table_node);
  NodeDef* bias_table_node = FindMutableNode(graph, info->bias_table_node);

  if ((add_table_node == nullptr || bias_table_node == nullptr) &&
      table_device.empty()) {
    for (const auto& node : graph->node()) {
      if (!BelongsToGroup(node.name(), group)) {
        continue;
      }
      if (!node.device().empty()) {
        table_device = node.device();
        break;
      }
    }
  }

  if (add_table_node == nullptr) {
    AddFloatTableConstNode(graph, info->add_table_node, table_device,
                           info->add_table_values, rows, info->row_width);
    add_table_node = FindMutableNode(graph, info->add_table_node);
  }
  if (bias_table_node == nullptr) {
    AddFloatTableConstNode(graph, info->bias_table_node, table_device,
                           info->bias_table_values, rows, info->row_width);
    bias_table_node = FindMutableNode(graph, info->bias_table_node);
  }

  if (!add_table_node || !bias_table_node || !IsOp(*add_table_node, "Const") ||
      !IsOp(*bias_table_node, "Const")) {
    info->enabled = false;
    return false;
  }

  // Keep const nodes synchronized with cache. Rewriting on append guarantees
  // existing table_index values remain stable while allowing incremental
  // growth.
  if (appended_new_rows ||
      add_table_node->attr().find("value") == add_table_node->attr().end() ||
      bias_table_node->attr().find("value") == bias_table_node->attr().end()) {
    UpdateFloatTableConstNode(add_table_node, info->add_table_values, rows,
                              info->row_width);
    UpdateFloatTableConstNode(bias_table_node, info->bias_table_values, rows,
                              info->row_width);
  }

  info->enabled = true;

  VLOG(1) << "MusaPlnCascadeFusion: table ready for group '" << group
          << "' with " << rows << " row(s), width=" << info->row_width
          << ", appended_new_rows=" << appended_new_rows;
  return true;
}

Status MusaPlnCascadeFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid PlnCascade match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  // Fusion pattern instances are reused across optimizer runs. Reset table
  // cache when the graph object changes to avoid leaking stale mappings.
  if (cache_graph_ != graph) {
    group_table_cache_.clear();
    cache_graph_ = graph;
  }

  auto select_it = match_result.captured_nodes.find("select");
  auto affine_it = match_result.captured_nodes.find("affine");
  if (select_it == match_result.captured_nodes.end() ||
      affine_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured select/affine node for PlnCascade");
  }

  const std::string select_name = select_it->second->name();
  const std::string affine_name = affine_it->second->name();

  auto select_on_true_it = match_result.captured_attrs.find("select_on_true");
  if (select_on_true_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing select_on_true attr in PlnCascade match result");
  }
  const bool select_on_true = (select_on_true_it->second == "1");

  const bool is_pending_affine =
      (match_result.captured_attrs.count("is_pending_affine") > 0 &&
       match_result.captured_attrs.at("is_pending_affine") == "1");

  std::string expected_affine_name = affine_name;
  auto affine_name_it = match_result.captured_attrs.find("affine_name");
  if (affine_name_it != match_result.captured_attrs.end() &&
      !affine_name_it->second.empty()) {
    expected_affine_name = affine_name_it->second;
  }

  auto group_it = match_result.captured_attrs.find("group");
  if (group_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing group attr in PlnCascade match result");
  }
  const std::string group = group_it->second;

  const NodeDef* select_node = FindNode(*graph, select_name);
  if (!select_node || !IsOp(*select_node, "Select") ||
      select_node->input_size() != 3) {
    return Status::OK();
  }

  const NodeDef* affine_node = FindProducer(
      *graph, select_on_true ? select_node->input(1) : select_node->input(2));
  if (!affine_node || affine_node->name() != expected_affine_name) {
    return Status::OK();
  }

  if (is_pending_affine) {
    if (!IsOp(*affine_node, "AddV2")) {
      return Status::OK();
    }
  } else {
    if (!IsOp(*affine_node, "MusaShiftedAffineMap") ||
        affine_node->input_size() != 3) {
      return Status::OK();
    }
  }

  std::string norm_input =
      select_on_true ? select_node->input(2) : select_node->input(1);
  auto norm_it = match_result.captured_attrs.find("norm_input");
  if (norm_it != match_result.captured_attrs.end() &&
      !norm_it->second.empty()) {
    norm_input = norm_it->second;
  }

  std::string add_input_edge;
  auto add_input_it = match_result.captured_attrs.find("add_input_edge");
  if (add_input_it != match_result.captured_attrs.end()) {
    add_input_edge = add_input_it->second;
  }
  std::string bias_input_edge;
  auto bias_input_it = match_result.captured_attrs.find("bias_input_edge");
  if (bias_input_it != match_result.captured_attrs.end()) {
    bias_input_edge = bias_input_it->second;
  }

  if (add_input_edge.empty() || bias_input_edge.empty()) {
    if (is_pending_affine) {
      return Status::OK();
    }
    add_input_edge = affine_node->input(0);
    bias_input_edge = affine_node->input(2);
  }

  std::string add_name;
  auto add_name_it = match_result.captured_attrs.find("add_name");
  if (add_name_it != match_result.captured_attrs.end()) {
    add_name = add_name_it->second;
  }
  if (add_name.empty()) {
    add_name = GetProducerName(add_input_edge);
  }

  std::string bias_name;
  auto bias_name_it = match_result.captured_attrs.find("bias_name");
  if (bias_name_it != match_result.captured_attrs.end()) {
    bias_name = bias_name_it->second;
  }
  if (bias_name.empty()) {
    bias_name = GetProducerName(bias_input_edge);
  }

  if (add_name.empty() || bias_name.empty()) {
    return Status::OK();
  }
  const std::string gate_input = select_node->input(0);

  bool use_table = false;
  int table_index = 0;

  GroupTableInfo& table_info = group_table_cache_[group];
  if (BuildGroupTables(graph, group, &table_info)) {
    const std::string key = MakePairKey(add_name, bias_name);
    auto idx_it = table_info.pair_to_index.find(key);
    if (idx_it != table_info.pair_to_index.end()) {
      use_table = true;
      table_index = idx_it->second;
      add_input_edge = table_info.add_table_node;
      bias_input_edge = table_info.bias_table_node;
    }
  }

  const int select_idx = FusionGraphUtils::FindNodeIndex(*graph, select_name);
  if (select_idx < 0) {
    return Status::OK();
  }
  const std::string output_device = select_node->device();
  FusionGraphUtils::RemoveNode(graph, select_idx);

  std::vector<std::string> removable_names;
  removable_names.push_back(affine_name);
  if (is_pending_affine) {
    auto mul_it = match_result.captured_attrs.find("pending_mul_name");
    if (mul_it != match_result.captured_attrs.end() &&
        !mul_it->second.empty()) {
      removable_names.push_back(mul_it->second);
    }
  }
  if (use_table) {
    removable_names.push_back(add_name);
    removable_names.push_back(bias_name);
    CollectBiasCleanupCandidates(*graph, bias_name, &removable_names);
  }
  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_names);

  NodeDef* fused = graph->add_node();
  fused->set_name(select_name);
  fused->set_op("MusaPlnCascade");
  fused->set_device(output_device);
  fused->add_input(norm_input);
  fused->add_input(gate_input);
  fused->add_input(add_input_edge);
  fused->add_input(bias_input_edge);

  (*fused->mutable_attr())["T"].set_type(DT_FLOAT);
  (*fused->mutable_attr())["use_table"].set_b(use_table);
  (*fused->mutable_attr())["table_index"].set_i(table_index);
  (*fused->mutable_attr())["select_on_true"].set_b(select_on_true);

  VLOG(1) << "MusaPlnCascadeFusion: fused select node '" << select_name
          << "' (use_table=" << use_table << ", table_index=" << table_index
          << ")";

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaPlnCascadeFusion);
REGISTER_FUSION_KERNEL(MusaPlnCascadeFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
