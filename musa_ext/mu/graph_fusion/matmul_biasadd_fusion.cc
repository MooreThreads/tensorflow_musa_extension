#include "mu/graph_fusion/matmul_biasadd_fusion.h"

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

std::string CanonicalizeInputName(const std::string& input) {
  if (input.empty()) return input;

  std::string node_name = input;
  if (node_name[0] == '^') {
    node_name = node_name.substr(1);
  }
  const size_t colon_pos = node_name.find(':');
  if (colon_pos != std::string::npos) {
    node_name = node_name.substr(0, colon_pos);
  }
  return node_name;
}

// Helper to find node's input producer
const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  if (input.empty()) return nullptr;

  const std::string node_name = CanonicalizeInputName(input);

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

bool MatMulBiasAddFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MatMulBiasAddFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& bias_add_node = graph.node(start_node_idx);

  // Start node must be BiasAdd / Add / AddV2
  if (!IsOp(bias_add_node, "BiasAdd")) {
    return result;
  }

  if (HasOriginalSuffix(bias_add_node.name())) {
    return result;
  }

  // Find MatMul node and bias node from the two inputs of BiasAdd/Add/AddV2
  const NodeDef* matmul_node = nullptr;
  const NodeDef* bias_node = nullptr;

  if (bias_add_node.input_size() >= 2) {
    const NodeDef* in0 = FindProducer(graph, bias_add_node.input(0));
    const NodeDef* in1 = FindProducer(graph, bias_add_node.input(1));

    if (in0 && IsOp(*in0, "MatMul")) {
      matmul_node = in0;
      bias_node = in1;
    } else if (in1 && IsOp(*in1, "MatMul")) {
      matmul_node = in1;
      bias_node = in0;
    }
  }

  if (!matmul_node || !bias_node) {
    return result;
  }

  // Record into result
  result.matched = true;
  result.matched_nodes.push_back(&bias_add_node);
  result.matched_nodes.push_back(matmul_node);

  result.captured_nodes["output"] = &bias_add_node;
  result.captured_nodes["bias_add"] = &bias_add_node;
  result.captured_nodes["matmul"] = matmul_node;
  result.captured_nodes["bias"] = bias_node;

  return result;
}

Status MatMulBiasAddFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid MatMulBiasAdd match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto bias_it = match_result.captured_nodes.find("bias");
  auto bias_add_it = match_result.captured_nodes.find("bias_add");

  if (output_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end() ||
      bias_it == match_result.captured_nodes.end() ||
      bias_add_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in MatMulBiasAdd pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* bias_node = bias_it->second;
  const NodeDef* bias_add_node = bias_add_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  // Avoid duplicate fusion
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaMatMulBiasAdd") {
      VLOG(1) << "MusaMatMulBiasAdd: Output node " << original_name
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

  VLOG(1) << "MatMulBiasAddFusion: Replacing " << original_name
          << " with MusaMatMulBiasAdd";

  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_output_node->device();

  // Pick dtype from MatMul first, then output node, otherwise float
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

  // Rename the original output node
  original_output_node->set_name(original_output_name);

  // Create fused node using the original output node name
  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaMatMulBiasAdd");
  fused_node->set_device(output_device);

  // MusaMatMulBiasAdd inputs: a, b, bias
  fused_node->add_input(matmul_node->input(0));
  fused_node->add_input(matmul_node->input(1));

  // Keep original bias edge string exactly, do not replace with
  // bias_node->name() because input could be "bias:0" instead of just "bias".
  const std::string input0_name = CanonicalizeInputName(bias_add_node->input(0));
  const std::string input1_name = CanonicalizeInputName(bias_add_node->input(1));
  const bool input0_is_matmul = input0_name == matmul_node->name();
  const bool input1_is_matmul = input1_name == matmul_node->name();

  if (input0_is_matmul == input1_is_matmul) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to determine bias input for MatMulBiasAdd fusion: " +
                      original_name);
  }

  fused_node->add_input(bias_add_node->input(input0_is_matmul ? 1 : 0));

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

  // Remove matched nodes if now unused.
  std::vector<std::string> removable_names = {original_output_name,
                                              matmul_node->name()};

  FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_names,
      {matmul_node->input(0), matmul_node->input(1), bias_node->name(),
       original_name});

  VLOG(1) << "MatMulBiasAddFusion: Successfully replaced '" << original_name
          << "' with MusaMatMulBiasAdd";

  return Status::OK();
}

// Register the pattern
REGISTER_FUSION_PATTERN(MatMulBiasAddFusion);

// Register kernel availability
REGISTER_FUSION_KERNEL(MatMulBiasAddFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow