#include "mu/graph_fusion/two_layer_fused_matmul_fusion.h"

#include <string>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"

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

bool IsSupportedFusionType(DataType dtype) {
  return dtype == DT_FLOAT || dtype == DT_DOUBLE || dtype == DT_HALF ||
         dtype == DT_BFLOAT16;
}

DataType ResolveNodeType(const NodeDef& node) {
  const auto dtype_it = node.attr().find("T");
  if (dtype_it != node.attr().end()) {
    return dtype_it->second.type();
  }
  return DT_INVALID;
}

bool GetBoolAttrOrDefault(const NodeDef& node, const char* attr_name,
                          bool default_value = false) {
  const auto attr_it = node.attr().find(attr_name);
  return attr_it != node.attr().end() ? attr_it->second.b() : default_value;
}

AttrValue ResolveOutputDType(const NodeDef* output_node,
                             const NodeDef* fallback_bias_add,
                             const NodeDef* fallback_matmul) {
  AttrValue dtype_attr;

  auto dtype_it = output_node->attr().find("T");
  if (dtype_it != output_node->attr().end()) {
    return dtype_it->second;
  }
  dtype_it = fallback_bias_add->attr().find("T");
  if (dtype_it != fallback_bias_add->attr().end()) {
    return dtype_it->second;
  }
  dtype_it = fallback_matmul->attr().find("T");
  if (dtype_it != fallback_matmul->attr().end()) {
    return dtype_it->second;
  }

  dtype_attr.set_type(DT_FLOAT);
  return dtype_attr;
}

void ProtectInputProducer(std::unordered_set<std::string>* protected_names,
                          const std::string& input_name) {
  if (!input_name.empty()) {
    protected_names->insert(FusionGraphUtils::GetProducerNodeName(input_name));
  }
}

struct MatMulBiasInputs {
  const NodeDef* matmul = nullptr;
  int bias_input_idx = -1;
};

MatMulBiasInputs ExtractMatMulFromBiasLike(const GraphDef& graph,
                                           const NodeDef& bias_like_node) {
  MatMulBiasInputs match;
  if (!IsBiasLike(bias_like_node) || bias_like_node.input_size() != 2) {
    return match;
  }

  const NodeDef* in0 = FindProducer(graph, bias_like_node.input(0));
  const NodeDef* in1 = FindProducer(graph, bias_like_node.input(1));
  if (in0 && IsOp(*in0, "MatMul")) {
    match.matmul = in0;
    match.bias_input_idx = 1;
  } else if (in1 && IsOp(*in1, "MatMul")) {
    match.matmul = in1;
    match.bias_input_idx = 0;
  }
  return match;
}

struct UniqueMatMulConsumerMatch {
  const NodeDef* matmul = nullptr;
  int producer_input_idx = -1;
};

UniqueMatMulConsumerMatch FindUniqueMatMulConsumer(const GraphDef& graph,
                                                   const NodeDef& producer) {
  UniqueMatMulConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& candidate = graph.node(i);
    if (candidate.name() == producer.name() ||
        HasOriginalSuffix(candidate.name())) {
      continue;
    }

    int producer_input_idx = -1;
    for (int j = 0; j < candidate.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(candidate.input(j)) ==
          producer.name()) {
        producer_input_idx = j;
        break;
      }
    }

    if (producer_input_idx < 0) continue;

    consumer_count++;
    if (consumer_count > 1) return {};

    if (!IsOp(candidate, "MatMul") || candidate.input_size() != 2) {
      return {};
    }

    match.matmul = &candidate;
    match.producer_input_idx = producer_input_idx;
  }

  return match;
}

struct UniqueBiasLikeConsumerMatch {
  const NodeDef* bias_add = nullptr;
  int bias_input_idx = -1;
};

UniqueBiasLikeConsumerMatch FindUniqueBiasLikeConsumer(const GraphDef& graph,
                                                       const NodeDef& producer) {
  UniqueBiasLikeConsumerMatch match;
  int consumer_count = 0;

  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& candidate = graph.node(i);
    if (candidate.name() == producer.name() ||
        HasOriginalSuffix(candidate.name())) {
      continue;
    }

    int producer_input_idx = -1;
    for (int j = 0; j < candidate.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(candidate.input(j)) ==
          producer.name()) {
        producer_input_idx = j;
        break;
      }
    }

    if (producer_input_idx < 0) continue;

    consumer_count++;
    if (consumer_count > 1) return {};

    if (!IsBiasLike(candidate) || candidate.input_size() != 2) {
      return {};
    }

    match.bias_add = &candidate;
    match.bias_input_idx = 1 - producer_input_idx;
  }

  return match;
}

}  // namespace

bool TwoLayerFusedMatMulFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult TwoLayerFusedMatMulFusion::Match(const GraphDef& graph,
                                                   int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& relu_node = graph.node(start_node_idx);
  if (!IsOp(relu_node, "Relu") || HasOriginalSuffix(relu_node.name()) ||
      relu_node.input_size() != 1) {
    return result;
  }

  const NodeDef* bias_add0 = FindProducer(graph, relu_node.input(0));
  if (!bias_add0 || !IsBiasLike(*bias_add0) ||
      !IsSupportedFusionType(ResolveNodeType(*bias_add0))) {
    return result;
  }

  const MatMulBiasInputs matmul_bias0 = ExtractMatMulFromBiasLike(graph, *bias_add0);
  if (matmul_bias0.matmul == nullptr || matmul_bias0.matmul->input_size() != 2) {
    return result;
  }

  const UniqueMatMulConsumerMatch matmul1_match =
      FindUniqueMatMulConsumer(graph, relu_node);
  if (matmul1_match.matmul == nullptr) {
    return result;
  }

  const UniqueBiasLikeConsumerMatch bias_add1_match =
      FindUniqueBiasLikeConsumer(graph, *matmul1_match.matmul);
  if (bias_add1_match.bias_add == nullptr ||
      !IsSupportedFusionType(ResolveNodeType(*bias_add1_match.bias_add))) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&relu_node, bias_add0, matmul_bias0.matmul,
                          matmul1_match.matmul, bias_add1_match.bias_add};
  result.captured_nodes["relu0"] = &relu_node;
  result.captured_nodes["bias_add0"] = bias_add0;
  result.captured_nodes["matmul0"] = matmul_bias0.matmul;
  result.captured_nodes["matmul1"] = matmul1_match.matmul;
  result.captured_nodes["bias_add1"] = bias_add1_match.bias_add;
  result.captured_attrs["input_a0"] = matmul_bias0.matmul->input(0);
  result.captured_attrs["input_b0"] = matmul_bias0.matmul->input(1);
  result.captured_attrs["bias0_input"] =
      bias_add0->input(matmul_bias0.bias_input_idx);
  result.captured_attrs["other1_input"] =
      matmul1_match.matmul->input(1 - matmul1_match.producer_input_idx);
  result.captured_attrs["bias1_input"] =
      bias_add1_match.bias_add->input(bias_add1_match.bias_input_idx);
  result.captured_attrs["hidden_input_idx1"] =
      std::to_string(matmul1_match.producer_input_idx);
  return result;
}

Status TwoLayerFusedMatMulFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid TwoLayerFusedMatMul match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  const auto bias_add0_it = match_result.captured_nodes.find("bias_add0");
  const auto matmul0_it = match_result.captured_nodes.find("matmul0");
  const auto relu0_it = match_result.captured_nodes.find("relu0");
  const auto matmul1_it = match_result.captured_nodes.find("matmul1");
  const auto bias_add1_it = match_result.captured_nodes.find("bias_add1");
  const auto input_a0_it = match_result.captured_attrs.find("input_a0");
  const auto input_b0_it = match_result.captured_attrs.find("input_b0");
  const auto bias0_input_it = match_result.captured_attrs.find("bias0_input");
  const auto other1_input_it = match_result.captured_attrs.find("other1_input");
  const auto bias1_input_it = match_result.captured_attrs.find("bias1_input");
  const auto hidden_input_idx1_it =
      match_result.captured_attrs.find("hidden_input_idx1");

  if (bias_add0_it == match_result.captured_nodes.end() ||
      matmul0_it == match_result.captured_nodes.end() ||
      relu0_it == match_result.captured_nodes.end() ||
      matmul1_it == match_result.captured_nodes.end() ||
      bias_add1_it == match_result.captured_nodes.end() ||
      input_a0_it == match_result.captured_attrs.end() ||
      input_b0_it == match_result.captured_attrs.end() ||
      bias0_input_it == match_result.captured_attrs.end() ||
      other1_input_it == match_result.captured_attrs.end() ||
      bias1_input_it == match_result.captured_attrs.end() ||
      hidden_input_idx1_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured nodes in TwoLayerFusedMatMul pattern");
  }

  const NodeDef* bias_add0 = bias_add0_it->second;
  const NodeDef* matmul0 = matmul0_it->second;
  const NodeDef* relu0 = relu0_it->second;
  const NodeDef* matmul1 = matmul1_it->second;
  const NodeDef* bias_add1 = bias_add1_it->second;

  const std::string fused_output_name = bias_add1->name();
  const std::string original_output_name = fused_output_name + "_original";
  const int output_node_idx =
      FusionGraphUtils::FindNodeIndex(*graph, fused_output_name);
  if (output_node_idx < 0) {
    return Status(error::INVALID_ARGUMENT,
                  "Failed to find output node in graph: " + fused_output_name);
  }

  for (const auto& node : graph->node()) {
    if (node.name() == fused_output_name &&
        node.op() == "MusaTwoLayerFusedMatMul") {
      return Status::OK();
    }
  }

  const bool transpose_a0 = GetBoolAttrOrDefault(*matmul0, "transpose_a");
  const bool transpose_b0 = GetBoolAttrOrDefault(*matmul0, "transpose_b");
  const bool transpose_a1 = GetBoolAttrOrDefault(*matmul1, "transpose_a");
  const bool transpose_b1 = GetBoolAttrOrDefault(*matmul1, "transpose_b");
  const int hidden_input_idx1 = std::stoi(hidden_input_idx1_it->second);
  const AttrValue dtype_attr =
      ResolveOutputDType(bias_add1, bias_add1, matmul1);

  NodeDef* mutable_output_node = graph->mutable_node(output_node_idx);
  const std::string output_device = mutable_output_node->device();
  mutable_output_node->set_name(original_output_name);

  NodeDef* fused_node = graph->add_node();
  fused_node->set_name(fused_output_name);
  fused_node->set_op("MusaTwoLayerFusedMatMul");
  fused_node->set_device(output_device);
  fused_node->add_input(input_a0_it->second);
  fused_node->add_input(input_b0_it->second);
  fused_node->add_input(bias0_input_it->second);
  fused_node->add_input(other1_input_it->second);
  fused_node->add_input(bias1_input_it->second);

  auto* attr = fused_node->mutable_attr();
  (*attr)["T"] = dtype_attr;
  (*attr)["transpose_a0"].set_b(transpose_a0);
  (*attr)["transpose_b0"].set_b(transpose_b0);
  (*attr)["transpose_a1"].set_b(transpose_a1);
  (*attr)["transpose_b1"].set_b(transpose_b1);
  (*attr)["hidden_input_idx1"].set_i(hidden_input_idx1);

  std::unordered_set<std::string> protected_node_names = {fused_output_name};
  ProtectInputProducer(&protected_node_names, input_a0_it->second);
  ProtectInputProducer(&protected_node_names, input_b0_it->second);
  ProtectInputProducer(&protected_node_names, bias0_input_it->second);
  ProtectInputProducer(&protected_node_names, other1_input_it->second);
  ProtectInputProducer(&protected_node_names, bias1_input_it->second);

  const std::vector<std::string> removable_nodes = {
      matmul0->name(), bias_add0->name(), relu0->name(), matmul1->name(),
      original_output_name};
  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_nodes,
                                        protected_node_names);
  return Status::OK();
}

REGISTER_FUSION_PATTERN(TwoLayerFusedMatMulFusion);
REGISTER_FUSION_KERNEL(TwoLayerFusedMatMulFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
