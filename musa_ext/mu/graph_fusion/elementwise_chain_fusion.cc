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

#include "mu/graph_fusion/elementwise_chain_fusion.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kernels/math/musa_fused_elementwise_kernel.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

struct StepSpec {
  int opcode = 0;
  int arity = 0;
  int arg_kind[::tensorflow::musa::kMusaFusedElementwiseMaxArity] = {
      ::tensorflow::musa::kOperandNone, ::tensorflow::musa::kOperandNone,
      ::tensorflow::musa::kOperandNone};
  int arg_input[::tensorflow::musa::kMusaFusedElementwiseMaxArity] = {-1, -1,
                                                                      -1};
};

struct ClusterBuildResult {
  bool valid = false;
  DataType dtype = DT_INVALID;
  std::vector<const NodeDef*> nodes;
  std::vector<std::string> data_inputs;
  std::vector<std::string> bool_inputs;
  std::vector<StepSpec> steps;
};

struct ClusterBuildState {
  explicit ClusterBuildState(DataType dtype_in) : dtype(dtype_in) {}

  const DataType dtype;
  std::vector<const NodeDef*> nodes;
  std::vector<std::string> data_inputs;
  std::vector<std::string> bool_inputs;
  std::vector<StepSpec> steps;
  std::unordered_map<std::string, int> data_input_map;
  std::unordered_map<std::string, int> bool_input_map;
  std::unordered_map<std::string, int> node_to_step;
  std::unordered_set<std::string> visiting;
};

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

bool HasControlInputs(const NodeDef& node) {
  for (int i = 0; i < node.input_size(); ++i) {
    if (!node.input(i).empty() && node.input(i)[0] == '^') {
      return true;
    }
  }
  return false;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string producer_name =
      FusionGraphUtils::GetProducerNodeName(input);
  if (producer_name.empty()) {
    return nullptr;
  }
  return FusionGraphUtils::GetNodeByName(graph, producer_name);
}

int CountConsumerNodes(const GraphDef& graph, const std::string& node_name) {
  std::unordered_set<std::string> consumer_names;
  for (const auto& node : graph.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      if (FusionGraphUtils::GetProducerNodeName(node.input(i)) == node_name) {
        consumer_names.insert(node.name());
      }
    }
  }
  return static_cast<int>(consumer_names.size());
}

bool GetNodeType(const NodeDef& node, DataType* dtype) {
  auto type_it = node.attr().find("T");
  if (type_it != node.attr().end()) {
    *dtype = type_it->second.type();
    return true;
  }
  type_it = node.attr().find("dtype");
  if (type_it != node.attr().end()) {
    *dtype = type_it->second.type();
    return true;
  }
  type_it = node.attr().find("value");
  if (type_it != node.attr().end()) {
    *dtype = type_it->second.tensor().dtype();
    return true;
  }
  return false;
}

bool IsSupportedDataType(DataType dtype) {
  return dtype == DT_FLOAT || dtype == DT_DOUBLE || dtype == DT_HALF ||
         dtype == DT_BFLOAT16;
}

bool GetSupportedOpSpec(const NodeDef& node, int* opcode, int* arity) {
  if (node.op() == "Add" || node.op() == "AddV2") {
    *opcode = ::tensorflow::musa::kOpcodeAdd;
    *arity = 2;
    return true;
  }
  if (node.op() == "Sub") {
    *opcode = ::tensorflow::musa::kOpcodeSub;
    *arity = 2;
    return true;
  }
  if (node.op() == "Mul") {
    *opcode = ::tensorflow::musa::kOpcodeMul;
    *arity = 2;
    return true;
  }
  if (node.op() == "RealDiv") {
    *opcode = ::tensorflow::musa::kOpcodeRealDiv;
    *arity = 2;
    return true;
  }
  if (node.op() == "Maximum") {
    *opcode = ::tensorflow::musa::kOpcodeMaximum;
    *arity = 2;
    return true;
  }
  if (node.op() == "Minimum") {
    *opcode = ::tensorflow::musa::kOpcodeMinimum;
    *arity = 2;
    return true;
  }
  if (node.op() == "Exp") {
    *opcode = ::tensorflow::musa::kOpcodeExp;
    *arity = 1;
    return true;
  }
  if (node.op() == "Log") {
    *opcode = ::tensorflow::musa::kOpcodeLog;
    *arity = 1;
    return true;
  }
  if (node.op() == "Rsqrt") {
    *opcode = ::tensorflow::musa::kOpcodeRsqrt;
    *arity = 1;
    return true;
  }
  if (node.op() == "Relu") {
    *opcode = ::tensorflow::musa::kOpcodeRelu;
    *arity = 1;
    return true;
  }
  if (node.op() == "Tanh") {
    *opcode = ::tensorflow::musa::kOpcodeTanh;
    *arity = 1;
    return true;
  }
  if (node.op() == "Sigmoid") {
    *opcode = ::tensorflow::musa::kOpcodeSigmoid;
    *arity = 1;
    return true;
  }
  if (node.op() == "Neg") {
    *opcode = ::tensorflow::musa::kOpcodeNeg;
    *arity = 1;
    return true;
  }
  if (node.op() == "Pow") {
    *opcode = ::tensorflow::musa::kOpcodePow;
    *arity = 2;
    return true;
  }
  if (node.op() == "Select" || node.op() == "SelectV2") {
    *opcode = ::tensorflow::musa::kOpcodeSelect;
    *arity = 3;
    return true;
  }
  return false;
}

bool IsSupportedElementwiseNode(const GraphDef& graph, const NodeDef& node,
                                DataType expected_dtype) {
  int opcode = 0;
  int arity = 0;
  if (!GetSupportedOpSpec(node, &opcode, &arity)) {
    return false;
  }
  if (!FusionGraphUtils::IsMusaNode(node)) {
    return false;
  }
  if (HasControlInputs(node) || HasOriginalSuffix(node.name())) {
    return false;
  }

  DataType dtype = DT_INVALID;
  if (!GetNodeType(node, &dtype) || !IsSupportedDataType(dtype)) {
    return false;
  }
  return dtype == expected_dtype;
}

int RegisterBoundaryInput(const std::string& input,
                          std::vector<std::string>* inputs,
                          std::unordered_map<std::string, int>* input_map) {
  const auto it = input_map->find(input);
  if (it != input_map->end()) {
    return it->second;
  }

  const int index = static_cast<int>(inputs->size());
  inputs->push_back(input);
  (*input_map)[input] = index;
  return index;
}

bool EncodeBoundaryOperand(const GraphDef& graph, const std::string& input_name,
                           DataType expected_dtype, ClusterBuildState* state,
                           int* operand_kind, int* operand_input) {
  const NodeDef* producer = FindProducer(graph, input_name);
  if (!producer) {
    return false;
  }

  DataType input_dtype = DT_INVALID;
  if (!GetNodeType(*producer, &input_dtype)) {
    return false;
  }

  if (input_dtype == DT_BOOL) {
    *operand_kind = ::tensorflow::musa::kOperandBoolInput;
    *operand_input =
        RegisterBoundaryInput(input_name, &state->bool_inputs,
                              &state->bool_input_map);
    return true;
  }

  if (input_dtype == expected_dtype) {
    *operand_kind = ::tensorflow::musa::kOperandDataInput;
    *operand_input =
        RegisterBoundaryInput(input_name, &state->data_inputs,
                              &state->data_input_map);
    return true;
  }

  return false;
}

bool BuildClusterRecursive(const GraphDef& graph, const NodeDef* node,
                           bool is_sink, ClusterBuildState* state) {
  const auto step_it = state->node_to_step.find(node->name());
  if (step_it != state->node_to_step.end()) {
    return true;
  }
  if (!IsSupportedElementwiseNode(graph, *node, state->dtype)) {
    return false;
  }
  if (!is_sink && CountConsumerNodes(graph, node->name()) != 1) {
    return false;
  }
  if (!state->visiting.insert(node->name()).second) {
    return false;
  }

  int opcode = 0;
  int arity = 0;
  if (!GetSupportedOpSpec(*node, &opcode, &arity) || node->input_size() != arity) {
    state->visiting.erase(node->name());
    return false;
  }

  StepSpec step;
  step.opcode = opcode;
  step.arity = arity;

  for (int input_idx = 0; input_idx < arity; ++input_idx) {
    const std::string input_name = node->input(input_idx);
    const NodeDef* producer = FindProducer(graph, input_name);
    const bool can_absorb =
        producer != nullptr &&
        IsSupportedElementwiseNode(graph, *producer, state->dtype) &&
        CountConsumerNodes(graph, producer->name()) == 1;

    if (can_absorb) {
      if (!BuildClusterRecursive(graph, producer, false, state)) {
        state->visiting.erase(node->name());
        return false;
      }
      const auto producer_it = state->node_to_step.find(producer->name());
      if (producer_it == state->node_to_step.end()) {
        state->visiting.erase(node->name());
        return false;
      }
      step.arg_kind[input_idx] = ::tensorflow::musa::kOperandStep;
      step.arg_input[input_idx] = producer_it->second;
      continue;
    }

    if (!EncodeBoundaryOperand(graph, input_name, state->dtype, state,
                               &step.arg_kind[input_idx],
                               &step.arg_input[input_idx])) {
      state->visiting.erase(node->name());
      return false;
    }
  }

  state->node_to_step[node->name()] = static_cast<int>(state->steps.size());
  state->steps.push_back(step);
  state->nodes.push_back(node);
  state->visiting.erase(node->name());
  return true;
}

bool IsHeavyOpcode(int opcode) {
  return opcode == ::tensorflow::musa::kOpcodeExp ||
         opcode == ::tensorflow::musa::kOpcodeLog ||
         opcode == ::tensorflow::musa::kOpcodeRsqrt ||
         opcode == ::tensorflow::musa::kOpcodeTanh ||
         opcode == ::tensorflow::musa::kOpcodeSigmoid ||
         opcode == ::tensorflow::musa::kOpcodePow ||
         opcode == ::tensorflow::musa::kOpcodeSelect;
}

bool HasInternalMerge(const std::vector<StepSpec>& steps) {
  for (const StepSpec& step : steps) {
    int internal_inputs = 0;
    for (int i = 0; i < step.arity; ++i) {
      if (step.arg_kind[i] == ::tensorflow::musa::kOperandStep) {
        ++internal_inputs;
      }
    }
    if (internal_inputs >= 2) {
      return true;
    }
  }
  return false;
}

int64_t GetStaticOutputElements(const NodeDef& node) {
  const auto output_shapes_it = node.attr().find("_output_shapes");
  if (output_shapes_it == node.attr().end() ||
      output_shapes_it->second.list().shape_size() <= 0) {
    return -1;
  }

  const auto& shape = output_shapes_it->second.list().shape(0);
  int64_t elements = 1;
  for (const auto& dim : shape.dim()) {
    if (dim.size() <= 0) {
      return -1;
    }
    elements *= dim.size();
  }
  return elements;
}

bool IsProfitableCluster(const ClusterBuildResult& cluster,
                         const NodeDef& sink_node) {
  const int num_steps = static_cast<int>(cluster.steps.size());
  if (num_steps < 2) {
    return false;
  }

  bool has_heavy_opcode = false;
  for (const StepSpec& step : cluster.steps) {
    if (IsHeavyOpcode(step.opcode)) {
      has_heavy_opcode = true;
      break;
    }
  }

  if (num_steps >= 4) {
    return true;
  }
  if (has_heavy_opcode && num_steps >= 2) {
    return true;
  }
  if (HasInternalMerge(cluster.steps) && num_steps >= 3) {
    return true;
  }

  const int64_t static_elements = GetStaticOutputElements(sink_node);
  return static_elements >= 4096 && num_steps >= 3;
}

ClusterBuildResult BuildElementwiseCluster(const GraphDef& graph,
                                           const NodeDef& sink) {
  ClusterBuildResult result;

  DataType dtype = DT_INVALID;
  if (!GetNodeType(sink, &dtype) || !IsSupportedDataType(dtype) ||
      !IsSupportedElementwiseNode(graph, sink, dtype)) {
    return result;
  }

  ClusterBuildState state(dtype);
  if (!BuildClusterRecursive(graph, &sink, true, &state)) {
    return result;
  }

  if (state.steps.size() > ::tensorflow::musa::kMusaFusedElementwiseMaxSteps ||
      state.data_inputs.size() >
          ::tensorflow::musa::kMusaFusedElementwiseMaxDataInputs ||
      state.bool_inputs.size() >
          ::tensorflow::musa::kMusaFusedElementwiseMaxBoolInputs) {
    return result;
  }

  result.valid = true;
  result.dtype = dtype;
  result.nodes = std::move(state.nodes);
  result.data_inputs = std::move(state.data_inputs);
  result.bool_inputs = std::move(state.bool_inputs);
  result.steps = std::move(state.steps);

  if (!IsProfitableCluster(result, sink)) {
    result.valid = false;
  }
  return result;
}

bool IsSinkCandidate(const GraphDef& graph, const NodeDef& node) {
  DataType dtype = DT_INVALID;
  if (!GetNodeType(node, &dtype) || !IsSupportedDataType(dtype) ||
      !IsSupportedElementwiseNode(graph, node, dtype)) {
    return false;
  }

  const int consumer_count = CountConsumerNodes(graph, node.name());
  if (consumer_count != 1) {
    return true;
  }

  for (const auto& consumer : graph.node()) {
    for (int i = 0; i < consumer.input_size(); ++i) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(i)) !=
          node.name()) {
        continue;
      }
      return !IsSupportedElementwiseNode(graph, consumer, dtype);
    }
  }
  return true;
}

void SetIntListAttr(NodeDef* node, const std::string& attr_name,
                    const std::vector<int>& values) {
  auto* list = (*node->mutable_attr())[attr_name].mutable_list();
  list->clear_i();
  for (int value : values) {
    list->add_i(value);
  }
}

}  // namespace

bool ElementwiseChainFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult ElementwiseChainFusion::Match(const GraphDef& graph,
                                                int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& node = graph.node(start_node_idx);
  if (!IsSinkCandidate(graph, node)) {
    return result;
  }

  const ClusterBuildResult cluster = BuildElementwiseCluster(graph, node);
  if (!cluster.valid) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = cluster.nodes;
  result.captured_nodes["sink"] = &node;
  result.captured_attrs["sink_name"] = node.name();
  return result;
}

Status ElementwiseChainFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid ElementwiseChainFusion match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const auto sink_name_it = match_result.captured_attrs.find("sink_name");
  if (sink_name_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing sink_name in ElementwiseChainFusion result");
  }

  const NodeDef* sink_node =
      FusionGraphUtils::GetNodeByName(*graph, sink_name_it->second);
  if (!sink_node) {
    return Status(error::INVALID_ARGUMENT,
                  "Sink node disappeared before apply");
  }

  const ClusterBuildResult cluster = BuildElementwiseCluster(*graph, *sink_node);
  if (!cluster.valid) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to rebuild elementwise cluster during apply");
  }

  const std::string original_name = sink_node->name();
  const std::string renamed_output_name = original_name + "_original";
  const std::string output_device = sink_node->device();
  const auto output_shapes_it = sink_node->attr().find("_output_shapes");
  const bool has_output_shapes = output_shapes_it != sink_node->attr().end();
  const AttrValue output_shapes_attr =
      has_output_shapes ? output_shapes_it->second : AttrValue();

  const int sink_node_idx = FusionGraphUtils::FindNodeIndex(*graph, original_name);
  if (sink_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate sink node for apply");
  }

  NodeDef* original_output_node = graph->mutable_node(sink_node_idx);
  original_output_node->set_name(renamed_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaFusedElementwise");
  fused_node->set_device(output_device);

  for (const auto& input : cluster.data_inputs) {
    fused_node->add_input(input);
  }
  for (const auto& input : cluster.bool_inputs) {
    fused_node->add_input(input);
  }

  (*fused_node->mutable_attr())["T"].set_type(cluster.dtype);
  (*fused_node->mutable_attr())["num_data_inputs"].set_i(
      static_cast<int64_t>(cluster.data_inputs.size()));
  (*fused_node->mutable_attr())["num_bool_inputs"].set_i(
      static_cast<int64_t>(cluster.bool_inputs.size()));

  std::vector<int> opcodes;
  std::vector<int> step_arities;
  std::vector<int> arg0_kinds;
  std::vector<int> arg0_inputs;
  std::vector<int> arg1_kinds;
  std::vector<int> arg1_inputs;
  std::vector<int> arg2_kinds;
  std::vector<int> arg2_inputs;

  opcodes.reserve(cluster.steps.size());
  step_arities.reserve(cluster.steps.size());
  arg0_kinds.reserve(cluster.steps.size());
  arg0_inputs.reserve(cluster.steps.size());
  arg1_kinds.reserve(cluster.steps.size());
  arg1_inputs.reserve(cluster.steps.size());
  arg2_kinds.reserve(cluster.steps.size());
  arg2_inputs.reserve(cluster.steps.size());

  for (const StepSpec& step : cluster.steps) {
    opcodes.push_back(step.opcode);
    step_arities.push_back(step.arity);
    arg0_kinds.push_back(step.arg_kind[0]);
    arg0_inputs.push_back(step.arg_input[0]);
    arg1_kinds.push_back(step.arg_kind[1]);
    arg1_inputs.push_back(step.arg_input[1]);
    arg2_kinds.push_back(step.arg_kind[2]);
    arg2_inputs.push_back(step.arg_input[2]);
  }

  SetIntListAttr(fused_node, "opcodes", opcodes);
  SetIntListAttr(fused_node, "step_arities", step_arities);
  SetIntListAttr(fused_node, "arg0_kinds", arg0_kinds);
  SetIntListAttr(fused_node, "arg0_inputs", arg0_inputs);
  SetIntListAttr(fused_node, "arg1_kinds", arg1_kinds);
  SetIntListAttr(fused_node, "arg1_inputs", arg1_inputs);
  SetIntListAttr(fused_node, "arg2_kinds", arg2_kinds);
  SetIntListAttr(fused_node, "arg2_inputs", arg2_inputs);

  if (has_output_shapes) {
    (*fused_node->mutable_attr())["_output_shapes"] = output_shapes_attr;
  }

  std::vector<std::string> removable_names;
  removable_names.reserve(cluster.nodes.size());
  removable_names.push_back(renamed_output_name);
  for (size_t i = 0; i + 1 < cluster.nodes.size(); ++i) {
    removable_names.push_back(cluster.nodes[i]->name());
  }

  std::unordered_set<std::string> protected_nodes = {original_name};
  for (const auto& input : cluster.data_inputs) {
    protected_nodes.insert(FusionGraphUtils::GetProducerNodeName(input));
  }
  for (const auto& input : cluster.bool_inputs) {
    protected_nodes.insert(FusionGraphUtils::GetProducerNodeName(input));
  }

  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_names, protected_nodes);
  return Status::OK();
}

REGISTER_FUSION_PATTERN(ElementwiseChainFusion);
REGISTER_FUSION_KERNEL(ElementwiseChainFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
