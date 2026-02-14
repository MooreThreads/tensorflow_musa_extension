#include "MusaGraphUtils_layout.h"

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace grappler {

NodeDef* MusaGraphUtils::CreateConstNode(GraphDef* graph, const string& name,
                                         const std::vector<int32>& values,
                                         const string& device) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Const");
  node->set_device(device);

  auto* attr = node->mutable_attr();
  (*attr)["dtype"].set_type(DT_INT32);
  auto* tensor = (*attr)["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->mutable_tensor_shape()->add_dim()->set_size(values.size());

  for (int32 v : values) {
    tensor->add_int_val(v);
  }
  return node;
}

NodeDef* MusaGraphUtils::InsertTranspose(GraphDef* graph,
                                         const string& base_name,
                                         const string& input_name,
                                         const std::vector<int32>& perm,
                                         DataType dtype, const string& device) {
  string perm_node_name = base_name + "/perm";
  CreateConstNode(graph, perm_node_name, perm, device);

  NodeDef* node = graph->add_node();
  node->set_name(base_name);
  node->set_op("Transpose");
  node->set_device(device);
  node->add_input(input_name);
  node->add_input(perm_node_name);

  auto* attr = node->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["Tperm"].set_type(DT_INT32);

  return node;
}

NodeDef* MusaGraphUtils::InsertCast(GraphDef* graph, const string& name,
                                    const string& input_name,
                                    DataType src_dtype, DataType dst_dtype,
                                    const string& device) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Cast");
  node->set_device(device);
  node->add_input(input_name);

  auto* attr = node->mutable_attr();
  (*attr)["SrcT"].set_type(src_dtype);
  (*attr)["DstT"].set_type(dst_dtype);
  (*attr)["Truncate"].set_b(false);

  return node;
}

void MusaGraphUtils::RedirectEdges(GraphDef* graph, const string& old_node_name,
                                   const string& new_node_name) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);

    if (node->name() == new_node_name) continue;

    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == old_node_name) {
        node->set_input(j, new_node_name);
      }
    }
  }
}

void MusaGraphUtils::RewriteLayoutAttributes(NodeDef* node) {
  auto* attr = node->mutable_attr();
  std::vector<string> layout_attrs = {"strides", "dilations"};

  for (const string& attr_name : layout_attrs) {
    if (attr->count(attr_name)) {
      auto* list = (*attr)[attr_name].mutable_list();
      if (list->i_size() == 4) {
        int64_t h = list->i(1);
        int64_t w = list->i(2);
        list->set_i(1, 1);
        list->set_i(2, h);
        list->set_i(3, w);
      }
    }
  }
}

bool MusaGraphUtils::IsMusaNCHWSupported(const NodeDef& node) {
  if (node.device().find("MUSA") == std::string::npos) return false;
  return kLayoutSensitiveOps(node) || kLayoutAgnosticOps(node);
}

bool MusaGraphUtils::kLayoutSensitiveOps(const NodeDef& node) {
  static const std::unordered_set<string> sensitive_ops = {
      "Conv2D",  "DepthwiseConv2dNative", "MaxPool",
      "AvgPool", "FusedBatchNorm",        "FusedBatchNormV3"};
  return sensitive_ops.count(node.op()) > 0;
}

bool MusaGraphUtils::kLayoutAgnosticOps(const NodeDef& node) {
  static const std::unordered_set<string> agnostic_ops = {
      "Relu", "Sigmoid", "Tanh", "BiasAdd", "Add", "Sub", "Mul", "Identity"};
  return agnostic_ops.count(node.op()) > 0;
}

void MusaGraphUtils::CleanupUnusedNodes(GraphDef* graph) {
  std::unordered_set<string> used_inputs;

  for (const auto& node : graph->node()) {
    for (const string& input : node.input()) {
      string node_name = input.substr(0, input.find(':'));
      if (node_name.find('^') == 0) node_name = node_name.substr(1);
      used_inputs.insert(node_name);
    }
  }

  GraphDef clean_graph;
  for (int i = 0; i < graph->node_size(); ++i) {
    const NodeDef& node = graph->node(i);
    if (used_inputs.count(node.name()) > 0 ||
        (node.op() != "Transpose" && node.op() != "Const")) {
      *clean_graph.add_node() = node;
    }
  }

  graph->Swap(&clean_graph);
}

}  // namespace grappler
}  // namespace tensorflow
