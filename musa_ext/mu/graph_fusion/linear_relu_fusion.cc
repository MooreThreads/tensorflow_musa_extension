#include "mu/graph_fusion/linear_relu_fusion.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

// Epsilon value for LayerNorm
constexpr float kDefaultEpsilon = 1e-6f;

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
  size_t colon_pos = node_name.find(':');
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

}  // namespace

// =============================================================================
// LinearReluFusion Implementation
// =============================================================================
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

  // find BiasAdd node
  const NodeDef* biasAdd_node = nullptr;

  for (int i = 0; i < relu_node.input_size(); ++i) {
    const NodeDef* input_node = FindProducer(graph, relu_node.input(i));
    if (!input_node) continue;

    if (IsOp(*input_node, "BiasAdd")) biasAdd_node = input_node;
  }

  if (!biasAdd_node) return result;

  // find Matmul node
  const NodeDef* matmul_node = nullptr;
  const NodeDef* bias_node = nullptr;
  for (int i = 0; i < biasAdd_node->input_size(); ++i) {
    const NodeDef* input_node = FindProducer(graph, biasAdd_node->input(i));
    if (!input_node) continue;

    if (IsOp(*input_node, "MatMul")) {
      matmul_node = input_node;
    } else {
      bias_node = input_node;
    }
  }

  if (!matmul_node) return result;

  // record into result
  result.matched = true;
  result.matched_nodes.push_back(&relu_node);
  result.matched_nodes.push_back(biasAdd_node);
  result.matched_nodes.push_back(matmul_node);

  result.captured_nodes["output"] = &relu_node;
  result.captured_nodes["bias_add"] = biasAdd_node;
  result.captured_nodes["matmul"] = matmul_node;
  if (bias_node) {
    result.captured_nodes["bias"] = bias_node;
  }

  // capture the inputs of Matmul
  for (int i = 0; i < matmul_node->input_size(); ++i) {
    const NodeDef* input_node = FindProducer(graph, matmul_node->input(i));
    if (input_node) {
      result.captured_nodes["input_" + std::to_string(i)] = input_node;
    }
  }

  return result;
}

Status LinearReluFusion::Apply(GraphDef* graph, const FusionMatchResult& match_result) const {
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
  
  if (output_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT, "Missing output node in LinearRelu pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* matmul_node = matmul_it->second;

  // create new LinearRelu node
  std::string fused_node_name = output_node->name() + "_linear_relu_fused";

  // Check if this output node has already been fused (avoid duplicates)
  // Extract the base name (remove trailing "_original" suffix if present)
  std::string base_name = output_node->name();
  if (base_name.size() > 9 && base_name.substr(base_name.size() - 9) == "_original") {
    base_name = base_name.substr(0, base_name.size() - 9);
  }
  
  // Check if there's already a linearRelu node with the base name
  for (const auto& node : graph->node()) {
    if (node.name() == base_name && node.op() == "MusaLinearRelu") {
      VLOG(1) << "MusaLinearRelu: Output node " << base_name 
              << " is already a fused node, skipping";
      return Status::OK();
    }
  }
  
  VLOG(1) << "LinearReluFusion: Creating fused node: " << fused_node_name;

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_node_name);
  fused_node->set_op("MusaLinearRelu");
  fused_node->set_device(output_node->device());

  // Set inputs for MusaLinearRelu: MatMul inputs then Bias input
  // 1. Add MatMul inputs
  for (int i = 0; i < matmul_node->input_size(); ++i) {
    fused_node->add_input(matmul_node->input(i));
  }
  // 2. Add Bias input
  if (bias_it != match_result.captured_nodes.end() && bias_it->second) {
    fused_node->add_input(match_result.captured_nodes.at("bias_add")->input(1));
  }

  // Copy essential attributes
  if (matmul_node->attr().count("transpose_a")) {
    (*fused_node->mutable_attr())["transpose_a"] = matmul_node->attr().at("transpose_a");
  }
  if (matmul_node->attr().count("transpose_b")) {
    (*fused_node->mutable_attr())["transpose_b"] = matmul_node->attr().at("transpose_b");
  }

  // Redirect all inputs from the output node to the fused node
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == fused_node_name) continue;
    
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == output_node->name()) {
        node->set_input(j, fused_node_name);
      } else if (node->input(j).find(output_node->name() + ":") == 0) {
        std::string suffix = node->input(j).substr(output_node->name().length());
        node->set_input(j, fused_node_name + suffix);
      }
    }
  }

  // Rename the original output node and give the fused node the original name
  // This ensures that direct fetches of the output tensor get the fused result
  std::string original_name = output_node->name();
  const_cast<NodeDef*>(output_node)->set_name(original_name + "_original");
  fused_node->set_name(original_name);
  
  // Also update any references to the renamed original node
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if (node->name() == original_name) continue;  // Skip the fused node (now has original name)
    
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == original_name + "_original") {
        // These should point to the fused node (which now has the original name)
        node->set_input(j, original_name);
      }
    }
  }
  
  VLOG(1) << "LinearReluFusion: Renamed fused node to " << original_name;
  
  return Status::OK();
}

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow