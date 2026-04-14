#include "mu/graph_fusion/two_layer_concat_matmul_fusion.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;
  const std::string node_name = FusionGraphUtils::GetProducerNodeName(input);
  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).name() == node_name) {
      return &graph.node(i);
    }
  }
  return nullptr;
}

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

void ProtectInputProducer(std::unordered_set<std::string>* protected_names,
                          const std::string& input_name) {
  if (!input_name.empty()) {
    protected_names->insert(FusionGraphUtils::GetProducerNodeName(input_name));
  }
}

struct UniqueMatMulConsumerMatch {
  const NodeDef* matmul = nullptr;
  int producer_input_idx = -1;
};

struct UniqueBiasAddConsumerMatch {
  const NodeDef* bias_add = nullptr;
  std::string bias_input;
};

UniqueMatMulConsumerMatch FindUniqueMatMulConsumer(const GraphDef& graph,
                                                   const NodeDef& producer) {
  UniqueMatMulConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (consumer.name() == producer.name() || HasOriginalSuffix(consumer.name())) {
      continue;
    }

    int producer_input_idx = -1;
    for (int j = 0; j < consumer.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(j)) ==
          producer.name()) {
        producer_input_idx = j;
        break;
      }
    }
    if (producer_input_idx < 0) {
      continue;
    }

    consumer_count++;
    if (consumer_count > 1) {
      return {};
    }
    if (!IsOp(consumer, "MatMul") || consumer.input_size() != 2) {
      return {};
    }

    match.matmul = &consumer;
    match.producer_input_idx = producer_input_idx;
  }

  return match;
}

UniqueBiasAddConsumerMatch FindUniqueBiasAddConsumer(const GraphDef& graph,
                                                     const NodeDef& producer) {
  UniqueBiasAddConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (consumer.name() == producer.name() || HasOriginalSuffix(consumer.name())) {
      continue;
    }

    bool consumes_producer = false;
    for (int j = 0; j < consumer.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(j)) ==
          producer.name()) {
        consumes_producer = true;
        break;
      }
    }
    if (!consumes_producer) {
      continue;
    }

    consumer_count++;
    if (consumer_count > 1) {
      return {};
    }
    if (!IsOp(consumer, "BiasAdd") || consumer.input_size() != 2) {
      return {};
    }

    if (FusionGraphUtils::GetProducerNodeName(consumer.input(0)) ==
        producer.name()) {
      match.bias_add = &consumer;
      match.bias_input = consumer.input(1);
    } else if (FusionGraphUtils::GetProducerNodeName(consumer.input(1)) ==
               producer.name()) {
      match.bias_add = &consumer;
      match.bias_input = consumer.input(0);
    } else {
      return {};
    }
  }

  return match;
}

}  // namespace

bool TwoLayerConcatMatMulFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult TwoLayerConcatMatMulFusion::Match(
    const GraphDef& graph, int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& activation_node = graph.node(start_node_idx);
  std::string activation_type;
  float activation_alpha = 0.2f;
  if (IsOp(activation_node, "Relu")) {
    activation_type = "Relu";
  } else if (IsOp(activation_node, "LeakyRelu")) {
    activation_type = "LeakyRelu";
    const auto alpha_it = activation_node.attr().find("alpha");
    if (alpha_it != activation_node.attr().end()) {
      activation_alpha = alpha_it->second.f();
    }
  } else {
    return result;
  }

  if (HasOriginalSuffix(activation_node.name()) || activation_node.input_size() != 1) {
    return result;
  }

  const NodeDef* bias_add0 = FindProducer(graph, activation_node.input(0));
  if (!bias_add0 || !IsOp(*bias_add0, "BiasAdd") || bias_add0->input_size() != 2 ||
      HasOriginalSuffix(bias_add0->name())) {
    return result;
  }

  const NodeDef* in0 = FindProducer(graph, bias_add0->input(0));
  const NodeDef* in1 = FindProducer(graph, bias_add0->input(1));
  const NodeDef* matmul0 = nullptr;
  int bias0_input_idx = -1;
  if (in0 && IsOp(*in0, "MatMul")) {
    matmul0 = in0;
    bias0_input_idx = 1;
  } else if (in1 && IsOp(*in1, "MatMul")) {
    matmul0 = in1;
    bias0_input_idx = 0;
  }
  if (!matmul0 || HasOriginalSuffix(matmul0->name()) || matmul0->input_size() != 2) {
    return result;
  }

  const NodeDef* concat_node = nullptr;
  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* producer = FindProducer(graph, matmul0->input(i));
    if (producer && IsOp(*producer, "ConcatV2")) {
      concat_node = producer;
      concat_input_idx = i;
      break;
    }
  }
  if (!concat_node) {
    return result;
  }

  const UniqueMatMulConsumerMatch matmul1_match =
      FindUniqueMatMulConsumer(graph, activation_node);
  if (matmul1_match.matmul == nullptr) {
    return result;
  }

  const UniqueBiasAddConsumerMatch bias_add1_match =
      FindUniqueBiasAddConsumer(graph, *matmul1_match.matmul);
  if (bias_add1_match.bias_add == nullptr) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&activation_node, bias_add0, matmul0, concat_node,
                          matmul1_match.matmul, bias_add1_match.bias_add};
  result.captured_nodes["activation"] = &activation_node;
  result.captured_nodes["bias_add0"] = bias_add0;
  result.captured_nodes["matmul0"] = matmul0;
  result.captured_nodes["concat"] = concat_node;
  result.captured_nodes["matmul1"] = matmul1_match.matmul;
  result.captured_nodes["bias_add1"] = bias_add1_match.bias_add;
  result.captured_attrs["activation_type"] = activation_type;
  result.captured_attrs["input_b0"] = matmul0->input(1 - concat_input_idx);
  result.captured_attrs["bias0_input"] = bias_add0->input(bias0_input_idx);
  result.captured_attrs["other1_input"] =
      matmul1_match.matmul->input(1 - matmul1_match.producer_input_idx);
  result.captured_attrs["bias1_input"] = bias_add1_match.bias_input;
  result.captured_attrs["hidden_input_idx1"] =
      std::to_string(matmul1_match.producer_input_idx);
  if (activation_type == "LeakyRelu") {
    result.captured_attrs["activation_alpha"] =
        std::to_string(activation_alpha);
  }
  return result;
}

Status TwoLayerConcatMatMulFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid TwoLayerConcatMatMul match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const auto activation_it = match_result.captured_nodes.find("activation");
  const auto bias_add0_it = match_result.captured_nodes.find("bias_add0");
  const auto matmul0_it = match_result.captured_nodes.find("matmul0");
  const auto concat_it = match_result.captured_nodes.find("concat");
  const auto matmul1_it = match_result.captured_nodes.find("matmul1");
  const auto bias_add1_it = match_result.captured_nodes.find("bias_add1");
  const auto input_b0_it = match_result.captured_attrs.find("input_b0");
  const auto bias0_input_it = match_result.captured_attrs.find("bias0_input");
  const auto other1_input_it = match_result.captured_attrs.find("other1_input");
  const auto bias1_input_it = match_result.captured_attrs.find("bias1_input");
  const auto hidden_input_idx1_it =
      match_result.captured_attrs.find("hidden_input_idx1");
  const auto activation_type_it =
      match_result.captured_attrs.find("activation_type");
  if (activation_it == match_result.captured_nodes.end() ||
      bias_add0_it == match_result.captured_nodes.end() ||
      matmul0_it == match_result.captured_nodes.end() ||
      concat_it == match_result.captured_nodes.end() ||
      matmul1_it == match_result.captured_nodes.end() ||
      bias_add1_it == match_result.captured_nodes.end() ||
      input_b0_it == match_result.captured_attrs.end() ||
      bias0_input_it == match_result.captured_attrs.end() ||
      other1_input_it == match_result.captured_attrs.end() ||
      bias1_input_it == match_result.captured_attrs.end() ||
      hidden_input_idx1_it == match_result.captured_attrs.end() ||
      activation_type_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in TwoLayerConcatMatMul pattern");
  }

  const NodeDef* activation = activation_it->second;
  const NodeDef* bias_add0 = bias_add0_it->second;
  const NodeDef* matmul0 = matmul0_it->second;
  const NodeDef* concat_node = concat_it->second;
  const NodeDef* matmul1 = matmul1_it->second;
  const NodeDef* bias_add1 = bias_add1_it->second;
  const std::string input_b0 = input_b0_it->second;
  const std::string bias0_input = bias0_input_it->second;
  const std::string other1_input = other1_input_it->second;
  const std::string bias1_input = bias1_input_it->second;
  const std::string activation_type = activation_type_it->second;
  const float activation_alpha =
      match_result.captured_attrs.count("activation_alpha")
          ? std::stof(match_result.captured_attrs.at("activation_alpha"))
          : 0.2f;

  const std::string fused_output_name = bias_add1->name();
  const std::string original_matmul_name = matmul0->name() + "_original";
  const std::string original_output_name = fused_output_name + "_original";

  for (const auto& node : graph->node()) {
    if (node.name() == fused_output_name &&
        node.op() == "MusaTwoLayerConcatMatMul") {
      return Status::OK();
    }
  }

  int matmul0_node_idx = FusionGraphUtils::FindNodeIndex(*graph, matmul0->name());
  int output_node_idx = FusionGraphUtils::FindNodeIndex(*graph, fused_output_name);
  if (matmul0_node_idx < 0 || output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate nodes for TwoLayerConcatMatMul fusion");
  }

  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    if (FindProducer(*graph, matmul0->input(i)) == concat_node) {
      concat_input_idx = i;
      break;
    }
  }
  if (concat_input_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate Concat input in MatMul: " + matmul0->name());
  }

  const auto dtype_it = matmul0->attr().find("T");
  if (dtype_it == matmul0->attr().end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing T attr in MatMul node: " + matmul0->name());
  }

  const bool transpose_a0 = matmul0->attr().count("transpose_a")
                                ? matmul0->attr().at("transpose_a").b()
                                : false;
  const bool transpose_b0 = matmul0->attr().count("transpose_b")
                                ? matmul0->attr().at("transpose_b").b()
                                : false;
  const bool transpose_a1 = matmul1->attr().count("transpose_a")
                                ? matmul1->attr().at("transpose_a").b()
                                : false;
  const bool transpose_b1 = matmul1->attr().count("transpose_b")
                                ? matmul1->attr().at("transpose_b").b()
                                : false;
  const int num_concat_inputs = concat_node->input_size() - 1;
  std::vector<std::string> concat_inputs;
  concat_inputs.reserve(num_concat_inputs);
  for (int i = 0; i < num_concat_inputs; ++i) {
    concat_inputs.push_back(concat_node->input(i));
  }
  const std::string axis_input = concat_node->input(num_concat_inputs);

  graph->mutable_node(matmul0_node_idx)->set_name(original_matmul_name);
  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string device = original_output_node->device();
  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_output_name);
  fused_node->set_op("MusaTwoLayerConcatMatMul");
  fused_node->set_device(device);
  for (const auto& concat_input : concat_inputs) {
    fused_node->add_input(concat_input);
  }
  fused_node->add_input(axis_input);
  fused_node->add_input(input_b0);
  fused_node->add_input(bias0_input);
  fused_node->add_input(other1_input);
  fused_node->add_input(bias1_input);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = dtype_it->second;
  (*attr)["num_concat"].set_i(num_concat_inputs);
  (*attr)["concat_input_idx"].set_i(concat_input_idx);
  (*attr)["transpose_a0"].set_b(transpose_a0);
  (*attr)["transpose_b0"].set_b(transpose_b0);
  (*attr)["transpose_a1"].set_b(transpose_a1);
  (*attr)["transpose_b1"].set_b(transpose_b1);
  (*attr)["hidden_input_idx1"].set_i(std::stoi(hidden_input_idx1_it->second));
  (*attr)["activation_type"].set_s(activation_type);
  if (activation_type == "LeakyRelu") {
    (*attr)["activation_alpha"].set_f(activation_alpha);
  }

  std::unordered_set<std::string> protected_node_names = {fused_output_name};
  for (const auto& concat_input : concat_inputs) {
    ProtectInputProducer(&protected_node_names, concat_input);
  }
  ProtectInputProducer(&protected_node_names, axis_input);
  ProtectInputProducer(&protected_node_names, input_b0);
  ProtectInputProducer(&protected_node_names, bias0_input);
  ProtectInputProducer(&protected_node_names, other1_input);
  ProtectInputProducer(&protected_node_names, bias1_input);

  FusionGraphUtils::RemoveNodesIfUnused(
      graph,
      {original_matmul_name, concat_node->name(), bias_add0->name(),
       activation->name(), matmul1->name(), original_output_name},
      protected_node_names);

  return Status::OK();
}

REGISTER_FUSION_PATTERN(TwoLayerConcatMatMulFusion);
REGISTER_FUSION_KERNEL(TwoLayerConcatMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
