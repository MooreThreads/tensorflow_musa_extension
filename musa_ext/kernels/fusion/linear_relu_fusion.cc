#include "kernels/fusion/linear_relu_fusion.h"

#include <algorithm>
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

}  // namespace

bool LinearReluFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult LinearReluFusion::Match(const GraphDef& graph,
                                          int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& relu_node = graph.node(start_node_idx);

  // match start with relu node
  if (!IsOp(relu_node, "Relu")) return result;
  if (HasOriginalSuffix(relu_node.name())) return result;

  // find BiasAdd node
  const NodeDef* bias_add_node = nullptr;

  if (relu_node.input_size() > 0) {
    const NodeDef* input_node = FindProducer(graph, relu_node.input(0));
    if (input_node &&
        (IsOp(*input_node, "BiasAdd") || IsOp(*input_node, "Add") ||
         IsOp(*input_node, "AddV2"))) {
      bias_add_node = input_node;
    }
  }

  if (!bias_add_node) {
    return result;
  }

  // find Matmul node
  const NodeDef* matmul_node = nullptr;
  const NodeDef* bias_node = nullptr;

  if (bias_add_node->input_size() >= 2) {
    const NodeDef* in0 = FindProducer(graph, bias_add_node->input(0));
    const NodeDef* in1 = FindProducer(graph, bias_add_node->input(1));

    if (in0 && IsOp(*in0, "MatMul")) {
      matmul_node = in0;
      bias_node = in1;
    } else if (in1 && IsOp(*in1, "MatMul")) {
      matmul_node = in1;
      bias_node = in0;
    }
  }

  if (!matmul_node || !bias_node) return result;

  // record into result
  result.matched = true;
  result.matched_nodes.push_back(&relu_node);
  result.matched_nodes.push_back(bias_add_node);
  result.matched_nodes.push_back(matmul_node);

  result.captured_nodes["output"] = &relu_node;
  result.captured_nodes["bias_add"] = bias_add_node;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["bias"] = bias_node;

  return result;
}

Status LinearReluFusion::Apply(GraphDef* graph,
                               const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT, "Invalid LinearRelu match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  // Get captured nodes
  auto output_it = match_result.captured_nodes.find("output");
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto bias_it = match_result.captured_nodes.find("bias");

  if (output_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end() ||
      bias_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in LinearRelu pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* bias_node = bias_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  // Check if this output node has already been fused (avoid duplicates)
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaLinearRelu") {
      VLOG(1) << "MusaLinearRelu: Output node " << original_name
              << " is already a fused node, skipping";
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

  VLOG(1) << "LinearReluFusion: Replacing " << original_name
          << " with MusaLinearRelu";

  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_output_node->device();

  // Pick T from MatMul or BiasAdd or output
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
  fused_node->set_op("MusaLinearRelu");
  fused_node->set_device(output_device);

  // MusaLinearRelu inputs: a, b, bias
  fused_node->add_input(matmul_node->input(0));
  fused_node->add_input(matmul_node->input(1));
  // bias input might need port handling if it's more than just a name
  fused_node->add_input(
      match_result.captured_nodes.at("bias_add")
          ->input(match_result.captured_nodes.at("bias_add")->input(0) ==
                          matmul_node->name()
                      ? 1
                      : 0));

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = output_dtype;

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

  // Remove nodes if unused
  std::vector<std::string> removable_names = {
      original_output_name, match_result.captured_nodes.at("bias_add")->name(),
      matmul_node->name()};

  FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_names,
      {matmul_node->input(0), matmul_node->input(1), bias_node->name(),
       original_name});

  VLOG(1) << "LinearReluFusion: Successfully replaced '" << original_name
          << "' with MusaLinearRelu";

  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(LinearReluFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(LinearReluFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
