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

#include "mu/graph_fusion/pln_cascade_block_fusion.h"

#include <sstream>
#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

constexpr int kMaxBlockSteps = 16;

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

bool GetBoolAttr(const NodeDef& node, const std::string& attr_name,
                 bool default_value) {
  auto it = node.attr().find(attr_name);
  if (it == node.attr().end()) {
    return default_value;
  }
  return it->second.b();
}

bool GetIntAttr(const NodeDef& node, const std::string& attr_name, int* out) {
  if (!out) {
    return false;
  }
  auto it = node.attr().find(attr_name);
  if (it == node.attr().end()) {
    return false;
  }
  *out = static_cast<int>(it->second.i());
  return true;
}

bool GetIntListAttr(const NodeDef& node, const std::string& attr_name,
                    std::vector<int>* out) {
  if (!out) {
    return false;
  }

  auto it = node.attr().find(attr_name);
  if (it == node.attr().end()) {
    return false;
  }

  const auto& list = it->second.list();
  out->clear();
  out->reserve(list.i_size());
  for (int i = 0; i < list.i_size(); ++i) {
    out->push_back(static_cast<int>(list.i(i)));
  }
  return true;
}

bool GetBoolListAttr(const NodeDef& node, const std::string& attr_name,
                     std::vector<int>* out) {
  if (!out) {
    return false;
  }

  auto it = node.attr().find(attr_name);
  if (it == node.attr().end()) {
    return false;
  }

  const auto& list = it->second.list();
  out->clear();
  out->reserve(list.b_size());
  for (int i = 0; i < list.b_size(); ++i) {
    out->push_back(list.b(i) ? 1 : 0);
  }
  return true;
}

struct PlnNodeInfo {
  std::string node_name;
  std::string norm_input;
  std::string add_input;
  std::string bias_input;
  std::vector<std::string> gate_inputs;
  std::vector<int> table_indices;
  std::vector<int> select_on_true;
};

std::string JoinStrings(const std::vector<std::string>& values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ',';
    }
    oss << values[i];
  }
  return oss.str();
}

std::string JoinInts(const std::vector<int>& values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ',';
    }
    oss << values[i];
  }
  return oss.str();
}

std::vector<std::string> SplitStrings(const std::string& packed) {
  std::vector<std::string> parts;
  std::stringstream ss(packed);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      parts.push_back(item);
    }
  }
  return parts;
}

std::vector<int> SplitInts(const std::string& packed) {
  std::vector<int> values;
  std::stringstream ss(packed);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    values.push_back(std::stoi(item));
  }
  return values;
}

bool IsEligibleCascadeNode(const NodeDef& node) {
  if (!IsOp(node, "MusaPlnCascade") || node.input_size() != 4) {
    return false;
  }

  auto t_it = node.attr().find("T");
  if (t_it == node.attr().end() || t_it->second.type() != DT_FLOAT) {
    return false;
  }

  auto use_table_it = node.attr().find("use_table");
  if (use_table_it == node.attr().end() || !use_table_it->second.b()) {
    return false;
  }

  int table_index = 0;
  if (!GetIntAttr(node, "table_index", &table_index)) {
    return false;
  }

  return true;
}

bool IsEligibleBlockNode(const NodeDef& node) {
  if (!IsOp(node, "MusaPlnCascadeBlock") || node.input_size() < 4) {
    return false;
  }

  auto t_it = node.attr().find("T");
  if (t_it == node.attr().end() || t_it->second.type() != DT_FLOAT) {
    return false;
  }

  int n_steps = 0;
  if (!GetIntAttr(node, "N", &n_steps) || n_steps <= 0 ||
      n_steps > kMaxBlockSteps) {
    return false;
  }

  std::vector<int> table_indices;
  std::vector<int> select_on_true;
  if (!GetIntListAttr(node, "table_indices", &table_indices) ||
      !GetBoolListAttr(node, "select_on_true", &select_on_true)) {
    return false;
  }

  if (static_cast<int>(table_indices.size()) != n_steps ||
      static_cast<int>(select_on_true.size()) != n_steps) {
    return false;
  }

  return node.input_size() == (3 + n_steps);
}

bool IsEligiblePlnNode(const NodeDef& node) {
  return IsEligibleCascadeNode(node) || IsEligibleBlockNode(node);
}

bool ExtractNodeInfo(const NodeDef& node, PlnNodeInfo* info) {
  if (!info) {
    return false;
  }

  if (IsEligibleCascadeNode(node)) {
    int table_index = 0;
    if (!GetIntAttr(node, "table_index", &table_index)) {
      return false;
    }

    info->node_name = node.name();
    info->norm_input = node.input(0);
    info->add_input = node.input(2);
    info->bias_input = node.input(3);
    info->gate_inputs = {node.input(1)};
    info->table_indices = {table_index};
    info->select_on_true = {GetBoolAttr(node, "select_on_true", true) ? 1 : 0};
    return true;
  }

  if (IsEligibleBlockNode(node)) {
    int n_steps = 0;
    if (!GetIntAttr(node, "N", &n_steps)) {
      return false;
    }

    std::vector<int> table_indices;
    std::vector<int> select_on_true;
    if (!GetIntListAttr(node, "table_indices", &table_indices) ||
        !GetBoolListAttr(node, "select_on_true", &select_on_true)) {
      return false;
    }

    if (static_cast<int>(table_indices.size()) != n_steps ||
        static_cast<int>(select_on_true.size()) != n_steps ||
        node.input_size() != (3 + n_steps)) {
      return false;
    }

    info->node_name = node.name();
    info->norm_input = node.input(0);
    info->add_input = node.input(1);
    info->bias_input = node.input(2);
    info->gate_inputs.clear();
    info->gate_inputs.reserve(n_steps);
    for (int i = 0; i < n_steps; ++i) {
      info->gate_inputs.push_back(node.input(3 + i));
    }
    info->table_indices = std::move(table_indices);
    info->select_on_true = std::move(select_on_true);
    return true;
  }

  return false;
}

const NodeDef* FindSingleChainConsumer(const GraphDef& graph,
                                       const NodeDef& node,
                                       const std::string& group) {
  const int total_consumers = CountConsumers(graph, node.name());
  if (total_consumers != 1) {
    return nullptr;
  }

  for (const auto& candidate : graph.node()) {
    if (!IsEligiblePlnNode(candidate) || candidate.input_size() < 1) {
      continue;
    }
    if (GetProducerName(candidate.input(0)) != node.name()) {
      continue;
    }

    std::string candidate_group;
    if (!ExtractPlnGroup(candidate.name(), &candidate_group) ||
        candidate_group != group) {
      continue;
    }
    return &candidate;
  }

  return nullptr;
}

}  // namespace

MusaPlnCascadeBlockFusion::MusaPlnCascadeBlockFusion() = default;

bool MusaPlnCascadeBlockFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaPlnCascadeBlockFusion::Match(const GraphDef& graph,
                                                   int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& start = graph.node(start_node_idx);
  if (!IsEligiblePlnNode(start)) {
    return FusionMatchResult{};
  }

  PlnNodeInfo start_info;
  if (!ExtractNodeInfo(start, &start_info)) {
    return FusionMatchResult{};
  }

  std::string group;
  if (!ExtractPlnGroup(start.name(), &group)) {
    return FusionMatchResult{};
  }

  const NodeDef* prev = FindProducer(graph, start_info.norm_input);
  if (prev && IsEligiblePlnNode(*prev)) {
    std::string prev_group;
    if (ExtractPlnGroup(prev->name(), &prev_group) && prev_group == group) {
      return FusionMatchResult{};
    }
  }

  std::vector<const NodeDef*> chain;
  std::vector<PlnNodeInfo> chain_infos;
  chain.push_back(&start);
  chain_infos.push_back(start_info);

  const std::string add_input = start_info.add_input;
  const std::string bias_input = start_info.bias_input;
  int total_steps = static_cast<int>(start_info.table_indices.size());

  const NodeDef* current = &start;
  while (true) {
    const NodeDef* next = FindSingleChainConsumer(graph, *current, group);
    if (!next || !IsEligiblePlnNode(*next)) {
      break;
    }

    PlnNodeInfo next_info;
    if (!ExtractNodeInfo(*next, &next_info)) {
      break;
    }
    if (next_info.add_input != add_input ||
        next_info.bias_input != bias_input) {
      break;
    }

    const int next_steps = static_cast<int>(next_info.table_indices.size());
    if (next_steps <= 0 || total_steps + next_steps > kMaxBlockSteps) {
      break;
    }

    chain.push_back(next);
    chain_infos.push_back(next_info);
    total_steps += next_steps;
    current = next;
  }

  if (chain.size() < 2 || total_steps < 2) {
    return FusionMatchResult{};
  }

  std::vector<std::string> chain_names;
  std::vector<std::string> gate_inputs;
  std::vector<int> table_indices;
  std::vector<int> select_on_true;
  chain_names.reserve(chain_infos.size());
  gate_inputs.reserve(total_steps);
  table_indices.reserve(total_steps);
  select_on_true.reserve(total_steps);

  for (const auto& info : chain_infos) {
    chain_names.push_back(info.node_name);
    gate_inputs.insert(gate_inputs.end(), info.gate_inputs.begin(),
                       info.gate_inputs.end());
    table_indices.insert(table_indices.end(), info.table_indices.begin(),
                         info.table_indices.end());
    select_on_true.insert(select_on_true.end(), info.select_on_true.begin(),
                          info.select_on_true.end());
  }

  if (static_cast<int>(gate_inputs.size()) != total_steps ||
      static_cast<int>(table_indices.size()) != total_steps ||
      static_cast<int>(select_on_true.size()) != total_steps) {
    return FusionMatchResult{};
  }

  FusionMatchResult result;
  result.matched = true;
  result.matched_nodes.insert(result.matched_nodes.end(), chain.begin(),
                              chain.end());
  result.captured_nodes["head"] = chain.front();
  result.captured_nodes["tail"] = chain.back();

  result.captured_attrs["norm_input"] = chain_infos.front().norm_input;
  result.captured_attrs["add_input"] = add_input;
  result.captured_attrs["bias_input"] = bias_input;
  result.captured_attrs["num_steps"] = std::to_string(total_steps);
  result.captured_attrs["chain_names"] = JoinStrings(chain_names);
  result.captured_attrs["gate_inputs"] = JoinStrings(gate_inputs);
  result.captured_attrs["table_indices"] = JoinInts(table_indices);
  result.captured_attrs["select_on_true"] = JoinInts(select_on_true);

  return result;
}

Status MusaPlnCascadeBlockFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid PlnCascadeBlock match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto tail_it = match_result.captured_nodes.find("tail");
  if (tail_it == match_result.captured_nodes.end() || !tail_it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured tail node for PlnCascadeBlock");
  }

  const std::string output_name = tail_it->second->name();
  const NodeDef* tail_node = FindNode(*graph, output_name);
  if (!tail_node || !IsEligiblePlnNode(*tail_node)) {
    return Status::OK();
  }

  auto norm_it = match_result.captured_attrs.find("norm_input");
  auto add_it = match_result.captured_attrs.find("add_input");
  auto bias_it = match_result.captured_attrs.find("bias_input");
  auto gates_it = match_result.captured_attrs.find("gate_inputs");
  auto indices_it = match_result.captured_attrs.find("table_indices");
  auto select_it = match_result.captured_attrs.find("select_on_true");
  auto chain_it = match_result.captured_attrs.find("chain_names");
  if (norm_it == match_result.captured_attrs.end() ||
      add_it == match_result.captured_attrs.end() ||
      bias_it == match_result.captured_attrs.end() ||
      gates_it == match_result.captured_attrs.end() ||
      indices_it == match_result.captured_attrs.end() ||
      select_it == match_result.captured_attrs.end() ||
      chain_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured attrs for PlnCascadeBlock fusion");
  }

  std::vector<std::string> chain_names = SplitStrings(chain_it->second);
  std::vector<std::string> gate_inputs = SplitStrings(gates_it->second);
  std::vector<int> table_indices = SplitInts(indices_it->second);
  std::vector<int> select_on_true = SplitInts(select_it->second);

  const int num_steps = static_cast<int>(gate_inputs.size());
  if (num_steps < 2 || static_cast<int>(table_indices.size()) != num_steps ||
      static_cast<int>(select_on_true.size()) != num_steps ||
      num_steps > kMaxBlockSteps || chain_names.empty()) {
    return Status::OK();
  }

  DataType dtype = DT_FLOAT;
  auto dtype_it = tail_node->attr().find("T");
  if (dtype_it != tail_node->attr().end()) {
    dtype = dtype_it->second.type();
  }
  const std::string output_device = tail_node->device();

  const int tail_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (tail_idx < 0) {
    return Status::OK();
  }
  FusionGraphUtils::RemoveNode(graph, tail_idx);

  std::vector<std::string> removable_names;
  removable_names.reserve(chain_names.size());
  for (const auto& name : chain_names) {
    if (name != output_name) {
      removable_names.push_back(name);
    }
  }
  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_names);

  NodeDef* fused = graph->add_node();
  fused->set_name(output_name);
  fused->set_op("MusaPlnCascadeBlock");
  fused->set_device(output_device);
  fused->add_input(norm_it->second);
  fused->add_input(add_it->second);
  fused->add_input(bias_it->second);
  for (const auto& gate_input : gate_inputs) {
    fused->add_input(gate_input);
  }

  auto* attr = fused->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["N"].set_i(num_steps);

  auto* table_indices_attr = (*attr)["table_indices"].mutable_list();
  for (const int idx : table_indices) {
    table_indices_attr->add_i(idx);
  }

  auto* select_attr = (*attr)["select_on_true"].mutable_list();
  for (const int flag : select_on_true) {
    select_attr->add_b(flag != 0);
  }

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaPlnCascadeBlockFusion);
REGISTER_FUSION_KERNEL(MusaPlnCascadeBlockFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
