#include "mu/graph_fusion/concat_matmul_biasadd_fusion.h"

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

}  // namespace

bool ConcatMatMulBiasAddFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult ConcatMatMulBiasAddFusion::Match(const GraphDef& graph,
                                                   int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& bias_add_node = graph.node(start_node_idx);
  if (!IsOp(bias_add_node, "BiasAdd")) return result;
  if (HasOriginalSuffix(bias_add_node.name()) || bias_add_node.input_size() != 2) {
    return result;
  }

  const NodeDef* in0 = FindProducer(graph, bias_add_node.input(0));
  const NodeDef* in1 = FindProducer(graph, bias_add_node.input(1));

  const NodeDef* matmul_node = nullptr;
  int bias_input_idx = -1;
  if (in0 && IsOp(*in0, "MatMul")) {
    matmul_node = in0;
    bias_input_idx = 1;
  } else if (in1 && IsOp(*in1, "MatMul")) {
    matmul_node = in1;
    bias_input_idx = 0;
  }
  if (!matmul_node || HasOriginalSuffix(matmul_node->name()) ||
      matmul_node->input_size() != 2) {
    return result;
  }

  const NodeDef* concat_node = nullptr;
  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    const NodeDef* producer = FindProducer(graph, matmul_node->input(i));
    if (producer && IsOp(*producer, "ConcatV2")) {
      concat_node = producer;
      concat_input_idx = i;
      break;
    }
  }
  if (!concat_node) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&bias_add_node, matmul_node, concat_node};
  result.captured_nodes["output"] = &bias_add_node;
  result.captured_nodes["bias_add"] = &bias_add_node;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["concat"] = concat_node;
  result.captured_attrs["bias_input"] = bias_add_node.input(bias_input_idx);
  result.captured_attrs["concat_input_idx"] = std::to_string(concat_input_idx);
  return result;
}

Status ConcatMatMulBiasAddFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid ConcatMatMulBiasAdd match result");
  }
  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const auto output_it = match_result.captured_nodes.find("output");
  const auto bias_add_it = match_result.captured_nodes.find("bias_add");
  const auto matmul_it = match_result.captured_nodes.find("matmul");
  const auto concat_it = match_result.captured_nodes.find("concat");
  const auto bias_input_it = match_result.captured_attrs.find("bias_input");
  if (output_it == match_result.captured_nodes.end() ||
      bias_add_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end() ||
      concat_it == match_result.captured_nodes.end() ||
      bias_input_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in ConcatMatMulBiasAdd pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* bias_add_node = bias_add_it->second;
  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* concat_node = concat_it->second;
  const std::string bias_input = bias_input_it->second;

  const std::string fused_output_name = output_node->name();
  const std::string original_matmul_name = matmul_node->name() + "_original";
  const std::string original_output_name = fused_output_name + "_original";

  for (const auto& node : graph->node()) {
    if (node.name() == fused_output_name && node.op() == "MusaConcatMatMul") {
      return Status::OK();
    }
  }

  const auto dtype_it = matmul_node->attr().find("T");
  if (dtype_it == matmul_node->attr().end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing T attr in MatMul node: " + matmul_node->name());
  }

  int matmul_node_idx = FusionGraphUtils::FindNodeIndex(*graph, matmul_node->name());
  int output_node_idx = FusionGraphUtils::FindNodeIndex(*graph, fused_output_name);
  if (matmul_node_idx < 0 || output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate nodes for ConcatMatMulBiasAdd fusion");
  }

  int concat_input_idx = -1;
  for (int i = 0; i < 2; ++i) {
    if (FindProducer(*graph, matmul_node->input(i)) == concat_node) {
      concat_input_idx = i;
      break;
    }
  }
  if (concat_input_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to locate Concat input in MatMul: " +
                      matmul_node->name());
  }

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
  const std::string other_input = matmul_node->input(1 - concat_input_idx);

  graph->mutable_node(matmul_node_idx)->set_name(original_matmul_name);
  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string device = original_output_node->device();
  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_output_name);
  fused_node->set_op("MusaConcatMatMul");
  fused_node->set_device(device);
  for (const auto& concat_input : concat_inputs) {
    fused_node->add_input(concat_input);
  }
  fused_node->add_input(axis_input);
  fused_node->add_input(other_input);
  fused_node->add_input(bias_input);

  AttrValue fused_ops_attr;
  fused_ops_attr.mutable_list()->add_s("BiasAdd");
  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = dtype_it->second;
  (*attr)["transpose_a"].set_b(transpose_a);
  (*attr)["transpose_b"].set_b(transpose_b);
  (*attr)["num_concat"].set_i(num_concat_inputs);
  (*attr)["concat_input_idx"].set_i(concat_input_idx);
  (*attr)["fused_ops"] = fused_ops_attr;
  (*attr)["num_args"].set_i(1);

  std::unordered_set<std::string> protected_node_names = {fused_output_name};
  for (const auto& concat_input : concat_inputs) {
    ProtectInputProducer(&protected_node_names, concat_input);
  }
  ProtectInputProducer(&protected_node_names, axis_input);
  ProtectInputProducer(&protected_node_names, other_input);
  ProtectInputProducer(&protected_node_names, bias_input);

  FusionGraphUtils::RemoveNodesIfUnused(
      graph,
      {original_matmul_name, concat_node->name(), original_output_name},
      protected_node_names);

  return Status::OK();
}

REGISTER_FUSION_PATTERN(ConcatMatMulBiasAddFusion);
REGISTER_FUSION_KERNEL(ConcatMatMulBiasAddFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
