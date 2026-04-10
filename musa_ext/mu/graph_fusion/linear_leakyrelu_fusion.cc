#include "mu/graph_fusion/linear_leakyrelu_fusion.h"

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

bool IsBiasLike(const NodeDef& node) {
  return IsOp(node, "BiasAdd") || IsOp(node, "Add") || IsOp(node, "AddV2");
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

}  // namespace

bool LinearLeakyReluFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult LinearLeakyReluFusion::Match(const GraphDef& graph,
                                               int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& leaky_relu_node = graph.node(start_node_idx);
  if (!IsOp(leaky_relu_node, "LeakyRelu")) return result;
  if (HasOriginalSuffix(leaky_relu_node.name())) return result;

  const NodeDef* bias_add_node = nullptr;
  if (leaky_relu_node.input_size() > 0) {
    const NodeDef* input_node = FindProducer(graph, leaky_relu_node.input(0));
    if (input_node && IsBiasLike(*input_node)) {
      bias_add_node = input_node;
    }
  }
  if (!bias_add_node) return result;

  const NodeDef* matmul_node = nullptr;
  const NodeDef* bias_node = nullptr;
  int bias_input_idx = -1;
  if (bias_add_node->input_size() >= 2) {
    const NodeDef* in0 = FindProducer(graph, bias_add_node->input(0));
    const NodeDef* in1 = FindProducer(graph, bias_add_node->input(1));

    if (in0 && IsOp(*in0, "MatMul")) {
      matmul_node = in0;
      bias_node = in1;
      bias_input_idx = 1;
    } else if (in1 && IsOp(*in1, "MatMul")) {
      matmul_node = in1;
      bias_node = in0;
      bias_input_idx = 0;
    }
  }
  if (!matmul_node || !bias_node || bias_input_idx < 0) return result;

  result.matched = true;
  result.matched_nodes.push_back(&leaky_relu_node);
  result.matched_nodes.push_back(bias_add_node);
  result.matched_nodes.push_back(matmul_node);
  result.captured_nodes["output"] = &leaky_relu_node;
  result.captured_nodes["bias_add"] = bias_add_node;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["bias"] = bias_node;
  result.captured_attrs["bias_input"] = bias_add_node->input(bias_input_idx);
  const auto alpha_it = leaky_relu_node.attr().find("alpha");
  result.captured_attrs["alpha"] =
      std::to_string(alpha_it != leaky_relu_node.attr().end()
                         ? alpha_it->second.f()
                         : 0.2f);
  return result;
}

Status LinearLeakyReluFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid LinearLeakyRelu match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto bias_it = match_result.captured_nodes.find("bias");
  auto bias_add_it = match_result.captured_nodes.find("bias_add");
  auto bias_input_it = match_result.captured_attrs.find("bias_input");
  auto alpha_it = match_result.captured_attrs.find("alpha");

  if (output_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end() ||
      bias_it == match_result.captured_nodes.end() ||
      bias_add_it == match_result.captured_nodes.end() ||
      bias_input_it == match_result.captured_attrs.end() ||
      alpha_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in LinearLeakyRelu pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* bias_node = bias_it->second;
  const NodeDef* bias_add_node = bias_add_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaLinearLeakyRelu") {
      return Status::OK();
    }
  }

  int output_node_idx = -1;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == original_name) {
      output_node_idx = i;
      break;
    }
  }
  if (output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find output node in graph: " + original_name);
  }

  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_output_node->device();

  AttrValue output_dtype;
  auto dtype_it = matmul_node->attr().find("T");
  if (dtype_it != matmul_node->attr().end()) {
    output_dtype = dtype_it->second;
  } else {
    dtype_it = original_output_node->attr().find("T");
    if (dtype_it != original_output_node->attr().end()) {
      output_dtype = dtype_it->second;
    } else {
      output_dtype.set_type(DT_FLOAT);
    }
  }

  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaLinearLeakyRelu");
  fused_node->set_device(output_device);
  fused_node->add_input(matmul_node->input(0));
  fused_node->add_input(matmul_node->input(1));
  fused_node->add_input(bias_input_it->second);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = output_dtype;
  (*attr)["alpha"].set_f(std::stof(alpha_it->second));

  if (matmul_node->attr().count("transpose_a")) {
    (*attr)["transpose_a"] = matmul_node->attr().at("transpose_a");
  } else {
    (*attr)["transpose_a"].set_b(false);
  }
  if (matmul_node->attr().count("transpose_b")) {
    (*attr)["transpose_b"] = matmul_node->attr().at("transpose_b");
  } else {
    (*attr)["transpose_b"].set_b(false);
  }

  std::vector<std::string> removable_names = {
      original_output_name, bias_add_node->name(), matmul_node->name()};
  FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_names,
      {matmul_node->input(0), matmul_node->input(1), bias_node->name(),
       original_name});

  return Status::OK();
}

REGISTER_FUSION_PATTERN(LinearLeakyReluFusion);
REGISTER_FUSION_KERNEL(LinearLeakyReluFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
