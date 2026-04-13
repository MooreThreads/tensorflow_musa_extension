#include "mu/graph_fusion/concat_matmul_fusion.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  const size_t colon_pos = node_name.find(':');
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

bool HasOriginalSuffix(const std::string& node_name) {
  static const std::string kOriginalSuffix = "_original";
  return node_name.size() >= kOriginalSuffix.size() &&
         node_name.compare(node_name.size() - kOriginalSuffix.size(),
                           kOriginalSuffix.size(), kOriginalSuffix) == 0;
}

struct BiasAddConsumerMatch {
  const NodeDef* bias_add = nullptr;
  std::string bias_input;
};

struct ActivationConsumerMatch {
  const NodeDef* activation = nullptr;
  std::string activation_type;
  float activation_alpha = 0.2f;
};

struct UniqueMatMulConsumerMatch {
  const NodeDef* matmul = nullptr;
  int producer_input_idx = -1;
};

BiasAddConsumerMatch FindUniqueBiasAddConsumer(const GraphDef& graph,
                                               const NodeDef& producer) {
  BiasAddConsumerMatch match;
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

ActivationConsumerMatch FindUniqueActivationConsumer(const GraphDef& graph,
                                                     const NodeDef& bias_add) {
  ActivationConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (consumer.name() == bias_add.name() || HasOriginalSuffix(consumer.name())) {
      continue;
    }

    bool consumes_bias_add = false;
    for (int j = 0; j < consumer.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(j)) ==
          bias_add.name()) {
        consumes_bias_add = true;
        break;
      }
    }
    if (!consumes_bias_add) {
      continue;
    }

    consumer_count++;
    if (consumer_count > 1) {
      return {};
    }

    if (IsOp(consumer, "Relu")) {
      match.activation = &consumer;
      match.activation_type = "Relu";
      return match;
    }
    if (IsOp(consumer, "LeakyRelu")) {
      match.activation = &consumer;
      match.activation_type = "LeakyRelu";
      auto alpha_it = consumer.attr().find("alpha");
      if (alpha_it != consumer.attr().end()) {
        match.activation_alpha = alpha_it->second.f();
      }
      return match;
    }
    return {};
  }

  return match;
}

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

void ProtectInputProducer(std::unordered_set<std::string>* protected_names,
                          const std::string& input_name) {
  if (!input_name.empty()) {
    protected_names->insert(FusionGraphUtils::GetProducerNodeName(input_name));
  }
}

}  // namespace

bool ConcatMatMulFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult ConcatMatMulFusion::Match(const GraphDef& graph,
                                            int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& matmul_node = graph.node(start_node_idx);
  if (!IsOp(matmul_node, "MatMul")) return result;
  if (HasOriginalSuffix(matmul_node.name())) return result;

  const NodeDef* concat_node = nullptr;
  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    if (matmul_node.input_size() > i) {
      const NodeDef* input_node = FindProducer(graph, matmul_node.input(i));
      if (input_node && IsOp(*input_node, "ConcatV2")) {
        concat_node = input_node;
        concat_input_idx = i;
        break;
      }
    }
  }
  if (!concat_node) {
    return result;
  }

  result.matched = true;
  result.matched_nodes.push_back(&matmul_node);
  result.matched_nodes.push_back(concat_node);
  result.captured_nodes["matmul"] = &matmul_node;
  result.captured_nodes["concat"] = concat_node;
  result.captured_nodes["other_input"] =
      (concat_input_idx == 0) ? FindProducer(graph, matmul_node.input(1))
                              : FindProducer(graph, matmul_node.input(0));

  BiasAddConsumerMatch bias_match = FindUniqueBiasAddConsumer(graph, matmul_node);
  result.captured_attrs["with_bias"] = bias_match.bias_add ? "true" : "false";
  if (bias_match.bias_add == nullptr) {
    return result;
  }

  result.matched_nodes.push_back(bias_match.bias_add);
  result.captured_nodes["bias_add"] = bias_match.bias_add;
  result.captured_attrs["bias_input"] = bias_match.bias_input;

  ActivationConsumerMatch activation_match =
      FindUniqueActivationConsumer(graph, *bias_match.bias_add);
  if (activation_match.activation == nullptr) {
    return result;
  }

  result.matched_nodes.push_back(activation_match.activation);
  result.captured_nodes["activation"] = activation_match.activation;
  result.captured_attrs["activation_type"] = activation_match.activation_type;
  if (activation_match.activation_type == "LeakyRelu") {
    result.captured_attrs["activation_alpha"] =
        std::to_string(activation_match.activation_alpha);
  }

  const UniqueMatMulConsumerMatch second_matmul_match =
      FindUniqueMatMulConsumer(graph, *activation_match.activation);
  if (second_matmul_match.matmul == nullptr) {
    return result;
  }

  BiasAddConsumerMatch second_bias_match =
      FindUniqueBiasAddConsumer(graph, *second_matmul_match.matmul);
  if (second_bias_match.bias_add == nullptr) {
    return result;
  }

  result.matched_nodes.push_back(second_matmul_match.matmul);
  result.matched_nodes.push_back(second_bias_match.bias_add);
  result.captured_nodes["matmul1"] = second_matmul_match.matmul;
  result.captured_nodes["bias_add1"] = second_bias_match.bias_add;
  result.captured_attrs["other1_input"] =
      second_matmul_match.matmul->input(1 - second_matmul_match.producer_input_idx);
  result.captured_attrs["bias1_input"] = second_bias_match.bias_input;
  result.captured_attrs["hidden_input_idx1"] =
      std::to_string(second_matmul_match.producer_input_idx);

  return result;
}

Status ConcatMatMulFusion::Apply(GraphDef* graph,
                                 const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid ConcatMatMul match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto concat_it = match_result.captured_nodes.find("concat");
  auto bias_add_it = match_result.captured_nodes.find("bias_add");
  auto activation_it = match_result.captured_nodes.find("activation");
  auto matmul1_it = match_result.captured_nodes.find("matmul1");
  auto bias_add1_it = match_result.captured_nodes.find("bias_add1");
  auto with_bias_it = match_result.captured_attrs.find("with_bias");
  auto bias_input_it = match_result.captured_attrs.find("bias_input");
  auto activation_type_it = match_result.captured_attrs.find("activation_type");
  auto activation_alpha_it =
      match_result.captured_attrs.find("activation_alpha");
  auto other1_input_it = match_result.captured_attrs.find("other1_input");
  auto bias1_input_it = match_result.captured_attrs.find("bias1_input");
  auto hidden_input_idx1_it =
      match_result.captured_attrs.find("hidden_input_idx1");

  if (matmul_it == match_result.captured_nodes.end() ||
      concat_it == match_result.captured_nodes.end() ||
      with_bias_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in ConcatMatMul pattern");
  }

  const NodeDef* matmul0 = matmul_it->second;
  const NodeDef* concat_node = concat_it->second;
  const bool with_bias = with_bias_it->second == "true";
  const NodeDef* bias_add0 =
      bias_add_it != match_result.captured_nodes.end() ? bias_add_it->second
                                                       : nullptr;
  const NodeDef* activation =
      activation_it != match_result.captured_nodes.end() ? activation_it->second
                                                         : nullptr;
  const NodeDef* matmul1 =
      matmul1_it != match_result.captured_nodes.end() ? matmul1_it->second
                                                      : nullptr;
  const NodeDef* bias_add1 =
      bias_add1_it != match_result.captured_nodes.end() ? bias_add1_it->second
                                                        : nullptr;
  const std::string activation_type =
      activation_type_it != match_result.captured_attrs.end()
          ? activation_type_it->second
          : std::string();
  const float activation_alpha =
      activation_alpha_it != match_result.captured_attrs.end()
          ? std::stof(activation_alpha_it->second)
          : 0.2f;
  const bool with_activation = activation != nullptr;
  const bool with_two_layer =
      matmul1 != nullptr && bias_add1 != nullptr &&
      other1_input_it != match_result.captured_attrs.end() &&
      bias1_input_it != match_result.captured_attrs.end() &&
      hidden_input_idx1_it != match_result.captured_attrs.end();

  if (with_bias && (bias_add0 == nullptr ||
                    bias_input_it == match_result.captured_attrs.end())) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing BiasAdd capture in ConcatMatMul bias pattern");
  }

  const std::string fused_output_name =
      with_two_layer
          ? bias_add1->name()
          : (with_activation ? activation->name()
                             : (with_bias ? bias_add0->name() : matmul0->name()));
  const std::string original_matmul_name = matmul0->name() + "_original";
  const std::string original_output_name = fused_output_name + "_original";
  const auto dtype_it = matmul0->attr().find("T");
  if (dtype_it == matmul0->attr().end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing T attr in MatMul node: " + matmul0->name());
  }
  const AttrValue dtype_attr = dtype_it->second;
  const bool transpose_a0 = matmul0->attr().count("transpose_a")
                                ? matmul0->attr().at("transpose_a").b()
                                : false;
  const bool transpose_b0 = matmul0->attr().count("transpose_b")
                                ? matmul0->attr().at("transpose_b").b()
                                : false;
  const bool transpose_a1 =
      with_two_layer && matmul1->attr().count("transpose_a")
          ? matmul1->attr().at("transpose_a").b()
          : false;
  const bool transpose_b1 =
      with_two_layer && matmul1->attr().count("transpose_b")
          ? matmul1->attr().at("transpose_b").b()
          : false;

  const int num_concat_inputs = concat_node->input_size() - 1;
  std::vector<std::string> concat_inputs;
  concat_inputs.reserve(num_concat_inputs);
  for (int i = 0; i < num_concat_inputs; ++i) {
    concat_inputs.push_back(concat_node->input(i));
  }
  const std::string axis_input = concat_node->input(num_concat_inputs);

  int concat_in_matmul_idx = -1;
  for (int i = 0; i < 2; ++i) {
    if (FindProducer(*graph, matmul0->input(i)) == concat_node) {
      concat_in_matmul_idx = i;
      break;
    }
  }
  if (concat_in_matmul_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate Concat input in MatMul: " + matmul0->name());
  }

  const std::string other0_input = matmul0->input(1 - concat_in_matmul_idx);
  const std::string bias0_input =
      with_bias ? bias_input_it->second : std::string();
  const std::string other1_input =
      with_two_layer ? other1_input_it->second : std::string();
  const std::string bias1_input =
      with_two_layer ? bias1_input_it->second : std::string();

  for (const auto& node : graph->node()) {
    if (node.name() == fused_output_name &&
        (node.op() == "MusaConcatMatMul" ||
         node.op() == "MusaTwoLayerConcatMatMul")) {
      return Status::OK();
    }
  }

  int matmul0_node_idx = FusionGraphUtils::FindNodeIndex(*graph, matmul0->name());
  if (matmul0_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find MatMul node in graph: " + matmul0->name());
  }

  int output_node_idx = matmul0_node_idx;
  if (with_bias || with_activation || with_two_layer) {
    output_node_idx = FusionGraphUtils::FindNodeIndex(*graph, fused_output_name);
    if (output_node_idx < 0) {
      return Status(error::INVALID_ARGUMENT,
                    "Failed to find fused output node in graph: " +
                        fused_output_name);
    }
  }

  VLOG(1) << "ConcatMatMulFusion: Replacing " << fused_output_name << " with "
          << (with_two_layer ? "MusaTwoLayerConcatMatMul"
                             : "MusaConcatMatMul");

  graph->mutable_node(matmul0_node_idx)->set_name(original_matmul_name);
  NodeDef* output_node = graph->mutable_node(output_node_idx);
  const std::string device = output_node->device();
  output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_output_name);
  fused_node->set_op(with_two_layer ? "MusaTwoLayerConcatMatMul"
                                    : "MusaConcatMatMul");
  fused_node->set_device(device);
  for (const auto& concat_input : concat_inputs) {
    fused_node->add_input(concat_input);
  }
  fused_node->add_input(axis_input);
  fused_node->add_input(other0_input);
  if (with_two_layer) {
    fused_node->add_input(bias0_input);
    fused_node->add_input(other1_input);
    fused_node->add_input(bias1_input);
  } else if (with_bias) {
    fused_node->add_input(bias0_input);
  }

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = dtype_attr;
  (*attr)["num_concat"].set_i(num_concat_inputs);
  (*attr)["concat_input_idx"].set_i(concat_in_matmul_idx);

  if (with_two_layer) {
    (*attr)["transpose_a0"].set_b(transpose_a0);
    (*attr)["transpose_b0"].set_b(transpose_b0);
    (*attr)["transpose_a1"].set_b(transpose_a1);
    (*attr)["transpose_b1"].set_b(transpose_b1);
    (*attr)["hidden_input_idx1"].set_i(std::stoi(hidden_input_idx1_it->second));
    (*attr)["activation_type"].set_s(activation_type);
    if (activation_type == "LeakyRelu") {
      (*attr)["activation_alpha"].set_f(activation_alpha);
    }
  } else {
    (*attr)["transpose_a"].set_b(transpose_a0);
    (*attr)["transpose_b"].set_b(transpose_b0);
    if (with_bias) {
      AttrValue fused_ops_attr;
      fused_ops_attr.mutable_list()->add_s("BiasAdd");
      if (with_activation) {
        fused_ops_attr.mutable_list()->add_s(activation_type);
        if (activation_type == "LeakyRelu") {
          (*attr)["activation_alpha"].set_f(activation_alpha);
        }
      }
      (*attr)["fused_ops"] = fused_ops_attr;
      (*attr)["num_args"].set_i(1);
    } else {
      (*attr)["num_args"].set_i(0);
    }
  }

  std::unordered_set<std::string> protected_node_names = {fused_output_name};
  for (const auto& concat_input : concat_inputs) {
    ProtectInputProducer(&protected_node_names, concat_input);
  }
  ProtectInputProducer(&protected_node_names, axis_input);
  ProtectInputProducer(&protected_node_names, other0_input);
  if (with_bias) {
    ProtectInputProducer(&protected_node_names, bias0_input);
  }
  if (with_two_layer) {
    ProtectInputProducer(&protected_node_names, other1_input);
    ProtectInputProducer(&protected_node_names, bias1_input);
  }

  std::vector<std::string> removable_nodes = {original_matmul_name,
                                              concat_node->name()};
  if (with_bias) {
    removable_nodes.push_back(bias_add0->name());
  }
  if (with_two_layer) {
    removable_nodes.push_back(activation->name());
    removable_nodes.push_back(matmul1->name());
    removable_nodes.push_back(original_output_name);
  } else if (with_activation || with_bias) {
    removable_nodes.push_back(original_output_name);
  }

  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_nodes,
                                        protected_node_names);
  return Status::OK();
}

REGISTER_FUSION_PATTERN(ConcatMatMulFusion);
REGISTER_FUSION_KERNEL(ConcatMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
