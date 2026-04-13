#include "mu/graph_fusion/concat_matmul_fusion.h"

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

  // record into result
  result.matched = true;
  result.matched_nodes.push_back(&matmul_node);
  result.matched_nodes.push_back(concat_node);

  result.captured_nodes["matmul"] = &matmul_node;
  result.captured_nodes["concat"] = concat_node;
  result.captured_nodes["other_input"] =
      (concat_input_idx == 0) ? FindProducer(graph, matmul_node.input(1))
                              : FindProducer(graph, matmul_node.input(0));

  return result;
}

Status ConcatMatMulFusion::Apply(GraphDef* graph,
                                 const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return ::tsl::errors::InvalidArgument("Invalid ConcatMatMul match result");
  }

  if (!IsKernelAvailable()) {
    return ::tsl::OkStatus();
  }

  // Get captured nodes
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto concat_it = match_result.captured_nodes.find("concat");

  if (matmul_it == match_result.captured_nodes.end() ||
      concat_it == match_result.captured_nodes.end()) {
    return ::tsl::errors::InvalidArgument(
                  "Missing required nodes in ConcatMatMul pattern");
  }

  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* concat_node = concat_it->second;

  const std::string original_name = matmul_node->name();
  const std::string original_matmul_name = original_name + "_original";

  // Check if this node has already been fused
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaConcatMatMul") {
      return ::tsl::OkStatus();
    }
  }

  int matmul_node_idx = -1;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == original_name) {
      matmul_node_idx = i;
      break;
    }
  }

  if (matmul_node_idx < 0) {
    return ::tsl::errors::InvalidArgument(
                  "Failed to find MatMul node in graph: " + original_name);
  }

  VLOG(1) << "ConcatMatMulFusion: Replacing " << original_name
          << " with MusaConcatMatMul";

  NodeDef* matmul_node_mutable = graph->mutable_node(matmul_node_idx);
  const std::string device = matmul_node_mutable->device();

  matmul_node_mutable->set_name(original_matmul_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaConcatMatMul");
  fused_node->set_device(device);

  // MusaConcatMatMul inputs: Concat inputs..., axis, Other MatMul input
  int num_concat_inputs = concat_node->input_size() - 1;
  for (int i = 0; i < num_concat_inputs; ++i) {
    fused_node->add_input(concat_node->input(i));
  }
  // axis
  fused_node->add_input(concat_node->input(num_concat_inputs));

  // other matmul input
  int concat_in_matmul_idx = -1;
  for (int i = 0; i < 2; ++i) {
    if (FindProducer(*graph, matmul_node->input(i)) == concat_node) {
      concat_in_matmul_idx = i;
      break;
    }
  }
  fused_node->add_input(matmul_node->input(1 - concat_in_matmul_idx));

  // Attributes
  (*fused_node->mutable_attr())["T"] = matmul_node->attr().at("T");
  (*fused_node->mutable_attr())["transpose_a"] =
      matmul_node->attr().at("transpose_a");
  (*fused_node->mutable_attr())["transpose_b"] =
      matmul_node->attr().at("transpose_b");
  (*fused_node->mutable_attr())["num_concat"] = AttrValue();
  fused_node->mutable_attr()->at("num_concat").set_i(num_concat_inputs);
  (*fused_node->mutable_attr())["concat_input_idx"] = AttrValue();
  fused_node->mutable_attr()
      ->at("concat_input_idx")
      .set_i(concat_in_matmul_idx);

  return ::tsl::OkStatus();
}

REGISTER_FUSION_PATTERN(ConcatMatMulFusion);
REGISTER_FUSION_KERNEL(ConcatMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
