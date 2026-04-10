#include "mu/graph_fusion/concat_matmul_fusion.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Helper to check if node has specific op type
bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

// Helper to find node's input producer
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

BiasAddConsumerMatch FindUniqueBiasAddConsumer(const GraphDef& graph,
                                               const NodeDef& matmul_node) {
  BiasAddConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (consumer.name() == matmul_node.name() ||
        HasOriginalSuffix(consumer.name())) {
      continue;
    }

    bool consumes_matmul = false;
    for (int j = 0; j < consumer.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(consumer.input(j)) ==
          matmul_node.name()) {
        consumes_matmul = true;
        break;
      }
    }

    if (!consumes_matmul) {
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
        matmul_node.name()) {
      match.bias_add = &consumer;
      match.bias_input = consumer.input(1);
    } else if (FusionGraphUtils::GetProducerNodeName(consumer.input(1)) ==
               matmul_node.name()) {
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

}  // namespace

bool ConcatMatMulFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    // Check if MusaConcatMatMul op is registered
    kernel_available_ = true;  // Simplified for now
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

  // match start with MatMul node
  if (!IsOp(matmul_node, "MatMul")) return result;
  if (HasOriginalSuffix(matmul_node.name())) return result;

  // find ConcatV2 node as input 0 or 1
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

  BiasAddConsumerMatch bias_match = FindUniqueBiasAddConsumer(graph, matmul_node);
  // record into result
  result.matched = true;
  result.matched_nodes.push_back(&matmul_node);
  result.matched_nodes.push_back(concat_node);

  result.captured_nodes["matmul"] = &matmul_node;
  result.captured_nodes["concat"] = concat_node;
  result.captured_nodes["other_input"] =
      (concat_input_idx == 0) ? FindProducer(graph, matmul_node.input(1))
                              : FindProducer(graph, matmul_node.input(0));
  result.captured_attrs["with_bias"] = bias_match.bias_add ? "true" : "false";
  if (bias_match.bias_add != nullptr) {
    result.matched_nodes.push_back(bias_match.bias_add);
    result.captured_nodes["bias_add"] = bias_match.bias_add;
    result.captured_attrs["bias_input"] = bias_match.bias_input;

    ActivationConsumerMatch activation_match =
        FindUniqueActivationConsumer(graph, *bias_match.bias_add);
    if (activation_match.activation != nullptr) {
      result.matched_nodes.push_back(activation_match.activation);
      result.captured_nodes["activation"] = activation_match.activation;
      result.captured_attrs["activation_type"] =
          activation_match.activation_type;
      if (activation_match.activation_type == "LeakyRelu") {
        result.captured_attrs["activation_alpha"] =
            std::to_string(activation_match.activation_alpha);
      }
    }
  }

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

  // Get captured nodes
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto concat_it = match_result.captured_nodes.find("concat");
  auto bias_add_it = match_result.captured_nodes.find("bias_add");
  auto activation_it = match_result.captured_nodes.find("activation");
  auto with_bias_it = match_result.captured_attrs.find("with_bias");
  auto bias_input_it = match_result.captured_attrs.find("bias_input");
  auto activation_type_it =
      match_result.captured_attrs.find("activation_type");
  auto activation_alpha_it =
      match_result.captured_attrs.find("activation_alpha");

  if (matmul_it == match_result.captured_nodes.end() ||
      concat_it == match_result.captured_nodes.end() ||
      with_bias_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in ConcatMatMul pattern");
  }

  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* concat_node = concat_it->second;
  const bool with_bias = with_bias_it->second == "true";
  if (with_bias && (bias_add_it == match_result.captured_nodes.end() ||
                    bias_input_it == match_result.captured_attrs.end())) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing BiasAdd capture in ConcatMatMul bias pattern");
  }
  const NodeDef* bias_add_node =
      (bias_add_it != match_result.captured_nodes.end()) ? bias_add_it->second
                                                         : nullptr;
  const NodeDef* activation_node =
      (activation_it != match_result.captured_nodes.end()) ? activation_it->second
                                                           : nullptr;
  const std::string activation_type =
      (activation_type_it != match_result.captured_attrs.end())
          ? activation_type_it->second
          : std::string();
  const float activation_alpha =
      (activation_alpha_it != match_result.captured_attrs.end())
          ? std::stof(activation_alpha_it->second)
          : 0.2f;

  const bool with_activation = activation_node != nullptr;
  const std::string fused_output_name =
      with_activation ? activation_node->name()
                      : (with_bias ? bias_add_node->name() : matmul_node->name());
  const std::string concat_node_name = concat_node->name();
  const std::string original_matmul_name = matmul_node->name() + "_original";
  const std::string original_output_name = fused_output_name + "_original";
  const auto dtype_it = matmul_node->attr().find("T");
  if (dtype_it == matmul_node->attr().end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing T attr in MatMul node: " + matmul_node->name());
  }
  const AttrValue dtype_attr = dtype_it->second;
  const bool transpose_a = matmul_node->attr().count("transpose_a")
                               ? matmul_node->attr().at("transpose_a").b()
                               : false;
  const bool transpose_b = matmul_node->attr().count("transpose_b")
                               ? matmul_node->attr().at("transpose_b").b()
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
    if (FindProducer(*graph, matmul_node->input(i)) == concat_node) {
      concat_in_matmul_idx = i;
      break;
    }
  }
  if (concat_in_matmul_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate Concat input in MatMul: " +
                      matmul_node->name());
  }
  const std::string other_input = matmul_node->input(1 - concat_in_matmul_idx);
  const std::string bias_input =
      with_bias ? bias_input_it->second : std::string();
  std::unordered_set<std::string> protected_node_names = {fused_output_name};
  for (const auto& concat_input : concat_inputs) {
    protected_node_names.insert(
        FusionGraphUtils::GetProducerNodeName(concat_input));
  }
  protected_node_names.insert(FusionGraphUtils::GetProducerNodeName(axis_input));
  protected_node_names.insert(
      FusionGraphUtils::GetProducerNodeName(other_input));
  if (with_bias) {
    protected_node_names.insert(
        FusionGraphUtils::GetProducerNodeName(bias_input));
  }

  // Check if this node has already been fused
  for (const auto& node : graph->node()) {
    if (node.name() == fused_output_name && node.op() == "MusaConcatMatMul") {
      return Status::OK();
    }
  }

  int matmul_node_idx = -1;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == matmul_node->name()) {
      matmul_node_idx = i;
      break;
    }
  }

  if (matmul_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find MatMul node in graph: " + matmul_node->name());
  }

  int output_node_idx = matmul_node_idx;
  if (with_bias || with_activation) {
    output_node_idx = FusionGraphUtils::FindNodeIndex(*graph, fused_output_name);
    if (output_node_idx < 0) {
      return Status(error::INVALID_ARGUMENT,
                    "Failed to find fused output node in graph: " +
                        fused_output_name);
    }
  }

  VLOG(1) << "ConcatMatMulFusion: Replacing " << fused_output_name
          << " with MusaConcatMatMul";

  NodeDef* matmul_node_mutable = graph->mutable_node(matmul_node_idx);
  matmul_node_mutable->set_name(original_matmul_name);

  NodeDef* output_node_mutable = graph->mutable_node(output_node_idx);
  const std::string device = output_node_mutable->device();
  output_node_mutable->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_output_name);
  fused_node->set_op("MusaConcatMatMul");
  fused_node->set_device(device);

  // MusaConcatMatMul inputs: Concat inputs..., axis, Other MatMul input,
  // optional fused args...
  for (const auto& concat_input : concat_inputs) {
    fused_node->add_input(concat_input);
  }
  // axis
  fused_node->add_input(axis_input);

  // other matmul input
  fused_node->add_input(other_input);
  if (with_bias) {
    fused_node->add_input(bias_input);
  }

  // Attributes
  (*fused_node->mutable_attr())["T"] = dtype_attr;
  (*fused_node->mutable_attr())["transpose_a"].set_b(transpose_a);
  (*fused_node->mutable_attr())["transpose_b"].set_b(transpose_b);
  (*fused_node->mutable_attr())["num_concat"] = AttrValue();
  fused_node->mutable_attr()->at("num_concat").set_i(num_concat_inputs);
  (*fused_node->mutable_attr())["concat_input_idx"] = AttrValue();
  fused_node->mutable_attr()
      ->at("concat_input_idx")
      .set_i(concat_in_matmul_idx);
  if (with_bias) {
    AttrValue fused_ops_attr;
    fused_ops_attr.mutable_list()->add_s("BiasAdd");
    if (with_activation) {
      fused_ops_attr.mutable_list()->add_s(activation_type);
      if (activation_type == "LeakyRelu") {
        (*fused_node->mutable_attr())["activation_alpha"].set_f(
            activation_alpha);
      }
    }
    (*fused_node->mutable_attr())["fused_ops"] = fused_ops_attr;
    (*fused_node->mutable_attr())["num_args"].set_i(1);
  } else {
    (*fused_node->mutable_attr())["num_args"].set_i(0);
  }

  std::vector<std::string> removable_nodes = {original_matmul_name,
                                              concat_node_name};
  if (with_bias) {
    removable_nodes.push_back(bias_add_node->name());
  }
  if (with_activation) {
    removable_nodes.push_back(original_output_name);
  } else if (with_bias) {
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
