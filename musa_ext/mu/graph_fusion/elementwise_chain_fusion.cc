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
  int arg0_kind = ::tensorflow::musa::kOperandNone;
  int arg0_input = -1;
  int arg1_kind = ::tensorflow::musa::kOperandNone;
  int arg1_input = -1;
};

struct ChainBuildResult {
  bool valid = false;
  DataType dtype = DT_INVALID;
  std::vector<const NodeDef*> chain_nodes;
  std::vector<std::string> boundary_inputs;
  std::vector<StepSpec> steps;
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

int CountConsumers(const GraphDef& graph, const std::string& node_name) {
  int consumers = 0;
  for (const auto& node : graph.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      if (FusionGraphUtils::GetProducerNodeName(node.input(i)) == node_name) {
        ++consumers;
      }
    }
  }
  return consumers;
}

bool GetNodeType(const NodeDef& node, DataType* dtype) {
  auto type_it = node.attr().find("T");
  if (type_it == node.attr().end()) {
    return false;
  }
  *dtype = type_it->second.type();
  return true;
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
                          std::vector<std::string>* boundary_inputs,
                          std::unordered_map<std::string, int>* input_map) {
  auto it = input_map->find(input);
  if (it != input_map->end()) {
    return it->second;
  }

  const int index = static_cast<int>(boundary_inputs->size());
  boundary_inputs->push_back(input);
  (*input_map)[input] = index;
  return index;
}

bool MaybeSetOperandFromInput(int input_index, int internal_input_index,
                              const NodeDef& node,
                              std::vector<std::string>* boundary_inputs,
                              std::unordered_map<std::string, int>* input_map,
                              int* operand_kind, int* operand_input) {
  if (input_index == internal_input_index) {
    *operand_kind = ::tensorflow::musa::kOperandPrev;
    *operand_input = -1;
    return true;
  }

  *operand_kind = ::tensorflow::musa::kOperandInput;
  *operand_input = RegisterBoundaryInput(node.input(input_index), boundary_inputs,
                                         input_map);
  return true;
}

ChainBuildResult BuildLinearChain(const GraphDef& graph, const NodeDef& sink) {
  ChainBuildResult result;

  DataType dtype = DT_INVALID;
  if (!GetNodeType(sink, &dtype) || !IsSupportedDataType(dtype) ||
      !IsSupportedElementwiseNode(graph, sink, dtype)) {
    return result;
  }

  std::vector<const NodeDef*> reversed_nodes;
  std::vector<StepSpec> reversed_steps;
  std::vector<std::string> boundary_inputs;
  std::unordered_map<std::string, int> input_map;
  std::unordered_set<std::string> seen_nodes;

  const NodeDef* current = &sink;
  while (current != nullptr) {
    if (!seen_nodes.insert(current->name()).second) {
      return result;
    }

    int opcode = 0;
    int arity = 0;
    if (!GetSupportedOpSpec(*current, &opcode, &arity)) {
      return result;
    }
    if (current->input_size() != arity) {
      return result;
    }

    int candidate_count = 0;
    int internal_input_index = -1;
    for (int input_idx = 0; input_idx < current->input_size(); ++input_idx) {
      const NodeDef* producer = FindProducer(graph, current->input(input_idx));
      if (!producer ||
          !IsSupportedElementwiseNode(graph, *producer, dtype) ||
          CountConsumers(graph, producer->name()) != 1) {
        continue;
      }
      ++candidate_count;
      internal_input_index = input_idx;
    }

    if (candidate_count != 1) {
      internal_input_index = -1;
    }

    StepSpec step;
    step.opcode = opcode;
    step.arity = arity;
    if (arity == 1) {
      MaybeSetOperandFromInput(0, internal_input_index, *current,
                               &boundary_inputs, &input_map, &step.arg0_kind,
                               &step.arg0_input);
    } else if (arity == 2) {
      MaybeSetOperandFromInput(0, internal_input_index, *current,
                               &boundary_inputs, &input_map, &step.arg0_kind,
                               &step.arg0_input);
      MaybeSetOperandFromInput(1, internal_input_index, *current,
                               &boundary_inputs, &input_map, &step.arg1_kind,
                               &step.arg1_input);
    }

    reversed_nodes.push_back(current);
    reversed_steps.push_back(step);

    if (internal_input_index < 0) {
      break;
    }
    current = FindProducer(graph, current->input(internal_input_index));
  }

  if (reversed_nodes.size() < 2) {
    return result;
  }
  if (boundary_inputs.size() > ::tensorflow::musa::kMusaFusedElementwiseMaxInputs ||
      reversed_steps.size() > ::tensorflow::musa::kMusaFusedElementwiseMaxSteps) {
    return result;
  }

  result.valid = true;
  result.dtype = dtype;
  result.chain_nodes.assign(reversed_nodes.rbegin(), reversed_nodes.rend());
  result.steps.assign(reversed_steps.rbegin(), reversed_steps.rend());
  result.boundary_inputs = std::move(boundary_inputs);
  return result;
}

bool IsSinkCandidate(const GraphDef& graph, const NodeDef& node) {
  DataType dtype = DT_INVALID;
  if (!GetNodeType(node, &dtype) || !IsSupportedDataType(dtype) ||
      !IsSupportedElementwiseNode(graph, node, dtype)) {
    return false;
  }

  const int consumer_count = CountConsumers(graph, node.name());
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

  const ChainBuildResult chain = BuildLinearChain(graph, node);
  if (!chain.valid) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = chain.chain_nodes;
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

  auto sink_name_it = match_result.captured_attrs.find("sink_name");
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

  const ChainBuildResult chain = BuildLinearChain(*graph, *sink_node);
  if (!chain.valid) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to rebuild elementwise chain during apply");
  }

  std::vector<std::string> chain_node_names;
  chain_node_names.reserve(chain.chain_nodes.size());
  for (const NodeDef* node : chain.chain_nodes) {
    chain_node_names.push_back(node->name());
  }

  const std::string original_name = sink_node->name();
  const std::string renamed_output_name = original_name + "_original";
  const std::string output_device = sink_node->device();
  const auto output_shapes_it = sink_node->attr().find("_output_shapes");
  const bool has_output_shapes =
      output_shapes_it != sink_node->attr().end();
  const AttrValue output_shapes_attr =
      has_output_shapes ? output_shapes_it->second : AttrValue();

  int sink_node_idx = FusionGraphUtils::FindNodeIndex(*graph, original_name);
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

  for (const auto& input : chain.boundary_inputs) {
    fused_node->add_input(input);
  }

  (*fused_node->mutable_attr())["T"].set_type(chain.dtype);
  (*fused_node->mutable_attr())["num_inputs"].set_i(
      static_cast<int64_t>(chain.boundary_inputs.size()));

  std::vector<int> opcodes;
  std::vector<int> step_arities;
  std::vector<int> arg0_kinds;
  std::vector<int> arg0_inputs;
  std::vector<int> arg1_kinds;
  std::vector<int> arg1_inputs;

  opcodes.reserve(chain.steps.size());
  step_arities.reserve(chain.steps.size());
  arg0_kinds.reserve(chain.steps.size());
  arg0_inputs.reserve(chain.steps.size());
  arg1_kinds.reserve(chain.steps.size());
  arg1_inputs.reserve(chain.steps.size());

  for (const StepSpec& step : chain.steps) {
    opcodes.push_back(step.opcode);
    step_arities.push_back(step.arity);
    arg0_kinds.push_back(step.arg0_kind);
    arg0_inputs.push_back(step.arg0_input);
    arg1_kinds.push_back(step.arg1_kind);
    arg1_inputs.push_back(step.arg1_input);
  }

  SetIntListAttr(fused_node, "opcodes", opcodes);
  SetIntListAttr(fused_node, "step_arities", step_arities);
  SetIntListAttr(fused_node, "arg0_kinds", arg0_kinds);
  SetIntListAttr(fused_node, "arg0_inputs", arg0_inputs);
  SetIntListAttr(fused_node, "arg1_kinds", arg1_kinds);
  SetIntListAttr(fused_node, "arg1_inputs", arg1_inputs);

  if (has_output_shapes) {
    (*fused_node->mutable_attr())["_output_shapes"] = output_shapes_attr;
  }

  std::vector<std::string> removable_names;
  removable_names.reserve(chain_node_names.size());
  removable_names.push_back(renamed_output_name);
  for (size_t i = 0; i + 1 < chain_node_names.size(); ++i) {
    removable_names.push_back(chain_node_names[i]);
  }

  std::unordered_set<std::string> protected_nodes = {original_name};
  for (const auto& input : chain.boundary_inputs) {
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
