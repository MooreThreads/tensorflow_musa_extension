#include "kernels/fusion/RGProjection_fusion.h"

#include <algorithm>
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

}  // namespace

bool BiasAddReluMatMulFusion::IsKernelAvailable() const {
  if (IsDisabledByEnv()) {
    return false;
  }
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult BiasAddReluMatMulFusion::Match(const GraphDef& graph,
                                                 int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& matmul_node = graph.node(start_node_idx);

  // Start from final output node: MatMul
  if (!IsOp(matmul_node, "MatMul")) return result;
  if (HasOriginalSuffix(matmul_node.name())) return result;

  const NodeDef* relu_node = nullptr;
  const NodeDef* other_input_node = nullptr;
  int relu_input_slot = -1;

  // Find Relu input of MatMul
  if (matmul_node.input_size() >= 2) {
    const NodeDef* in0 = FindProducer(graph, matmul_node.input(0));
    const NodeDef* in1 = FindProducer(graph, matmul_node.input(1));

    if (in0 && IsOp(*in0, "Relu")) {
      relu_node = in0;
      other_input_node = in1;
      relu_input_slot = 0;
    } else if (in1 && IsOp(*in1, "Relu")) {
      relu_node = in1;
      other_input_node = in0;
      relu_input_slot = 1;
    }
  }

  if (!relu_node) return result;

  // Find BiasAdd feeding Relu
  const NodeDef* bias_add_node = nullptr;
  if (relu_node->input_size() > 0) {
    const NodeDef* input_node = FindProducer(graph, relu_node->input(0));
    if (input_node &&
        (IsOp(*input_node, "BiasAdd") || IsOp(*input_node, "Add") ||
         IsOp(*input_node, "AddV2"))) {
      bias_add_node = input_node;
    }
  }

  if (!bias_add_node) return result;

  // Find bias input of BiasAdd
  const NodeDef* bias_node = nullptr;
  if (bias_add_node->input_size() >= 2) {
    const NodeDef* in0 = FindProducer(graph, bias_add_node->input(0));
    const NodeDef* in1 = FindProducer(graph, bias_add_node->input(1));

    // For BiasAdd(x, b), typically bias is input(1),
    // but keep the logic a bit tolerant.
    if (in0 && !in1) {
      bias_node = in0;
    } else if (!in0 && in1) {
      bias_node = in1;
    } else if (in1) {
      bias_node = in1;
    } else if (in0) {
      bias_node = in0;
    }
  }

  // record into result, same style as linear_relu_fusion.cc
  result.matched = true;
  result.matched_nodes.push_back(&matmul_node);
  result.matched_nodes.push_back(relu_node);
  result.matched_nodes.push_back(bias_add_node);

  result.captured_nodes["output"] = &matmul_node;
  result.captured_nodes["matmul"] = &matmul_node;
  result.captured_nodes["relu"] = relu_node;
  result.captured_nodes["bias_add"] = bias_add_node;

  if (bias_node) {
    result.captured_nodes["bias"] = bias_node;
  }
  if (other_input_node) {
    result.captured_nodes["matmul_other_input"] = other_input_node;
  }

  // Ensure Relu and BiasAdd don't have multiple consumers preventing Safe
  // Fusion
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& consumer = graph.node(i);
    if (consumer.name() == matmul_node.name()) {
      continue;
    }
    for (int j = 0; j < consumer.input_size(); ++j) {
      const std::string producer_name =
          FusionGraphUtils::GetProducerNodeName(consumer.input(j));
      if (producer_name == relu_node->name()) {
        // Relu has a consumer other than MatMul
        return FusionMatchResult();
      }
      if (producer_name == bias_add_node->name() &&
          consumer.name() != relu_node->name()) {
        // BiasAdd has a consumer other than Relu
        return FusionMatchResult();
      }
    }
  }

  return result;
}

Status BiasAddReluMatMulFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid BiasAddReluMatMul match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto output_it = match_result.captured_nodes.find("output");
  auto matmul_it = match_result.captured_nodes.find("matmul");
  auto relu_it = match_result.captured_nodes.find("relu");
  auto bias_add_it = match_result.captured_nodes.find("bias_add");

  if (output_it == match_result.captured_nodes.end() ||
      matmul_it == match_result.captured_nodes.end() ||
      relu_it == match_result.captured_nodes.end() ||
      bias_add_it == match_result.captured_nodes.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing required nodes in BiasAddReluMatMul pattern");
  }

  const NodeDef* output_node = output_it->second;
  const NodeDef* matmul_node = matmul_it->second;
  const NodeDef* relu_node = relu_it->second;
  const NodeDef* bias_add_node = bias_add_it->second;

  const std::string original_name = output_node->name();
  const std::string original_output_name = original_name + "_original";

  // Avoid duplicates
  for (const auto& node : graph->node()) {
    if (node.name() == original_name && node.op() == "MusaBiasAddReluMatMul") {
      VLOG(1) << "MusaBiasAddReluMatMul: Output node " << original_name
              << " is already fused, skipping";
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

  VLOG(1) << "BiasAddReluMatMulFusion: Replacing " << original_name
          << " with MusaBiasAddReluMatMul";

  NodeDef* original_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = original_output_node->device();

  AttrValue output_dtype;
  auto dtype_it = matmul_node->attr().find("T");
  if (dtype_it != matmul_node->attr().end()) {
    output_dtype = dtype_it->second;
  } else {
    dtype_it = bias_add_node->attr().find("T");
    if (dtype_it != bias_add_node->attr().end()) {
      output_dtype = dtype_it->second;
    } else {
      output_dtype.set_type(DT_FLOAT);
    }
  }

  original_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(original_name);
  fused_node->set_op("MusaBiasAddReluMatMul");
  fused_node->set_device(output_device);

  // Determine which MatMul input is relu, and the other input string
  int relu_input_slot = -1;
  if (matmul_node->input_size() >= 2) {
    const NodeDef* in0 = FindProducer(*graph, matmul_node->input(0));
    const NodeDef* in1 = FindProducer(*graph, matmul_node->input(1));

    if (in0 && in0->name() == relu_node->name()) {
      relu_input_slot = 0;
    } else if (in1 && in1->name() == relu_node->name()) {
      relu_input_slot = 1;
    }
  }

  if (relu_input_slot < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to determine Relu input slot of MatMul");
  }

  const std::string other_input =
      matmul_node->input(relu_input_slot == 0 ? 1 : 0);

  // MusaBiasAddReluMatMul inputs:
  // 0: bias_add input data
  // 1: bias
  // 2: matmul other input
  fused_node->add_input(bias_add_node->input(0));
  fused_node->add_input(bias_add_node->input(1));
  fused_node->add_input(other_input);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = output_dtype;
  (*attr)["relu_input_slot"].set_i(relu_input_slot);

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
      original_output_name,
      relu_node->name(),
      bias_add_node->name(),
  };

  FusionGraphUtils::RemoveNodesIfUnused(
      graph, removable_names,
      {bias_add_node->input(0), bias_add_node->input(1), other_input,
       original_name});

  VLOG(1) << "BiasAddReluMatMulFusion: Successfully replaced '" << original_name
          << "' with MusaBiasAddReluMatMul";

  return Status::OK();
}

REGISTER_FUSION_PATTERN(BiasAddReluMatMulFusion);
REGISTER_FUSION_KERNEL(BiasAddReluMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
