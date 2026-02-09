#include "musa_remapper_optimizer.h"
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

Node* MusaOptimizationPass::FindNode(Graph* graph, const std::string& name) {
  for (Node* n : graph->nodes()) {
    if (n->name() == name) return n;
  }
  return nullptr;
}

int MusaOptimizationPass::CountConsumers(Node* node) {
  int count = 0;
  for (const Edge* e : node->out_edges()) {
    if (!e->IsControlEdge()) count++;
  }
  return count;
}

Status MusaOptimizationPass::Run(const GraphOptimizationPassOptions& options) {
  static bool logged = false;
  if (!logged) {
    fprintf(stderr, ">>>>> [MUSA_DEBUG] Optimization Pass Loaded (MatMul+BiasAdd+Relu) <<<<<\n");
    logged = true;
  }

  if (options.graph == nullptr) return Status::OK();
  Graph* graph = options.graph->get();

  // 收集所有的 BiasAdd 节点作为潜在的融合起点
  std::vector<Node*> bias_add_nodes;
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == "BiasAdd") {
      bias_add_nodes.push_back(n);
    }
  }

  bool graph_changed = false;

  for (Node* bias_node : bias_add_nodes) {
    // --- 检查输入 0: 必须来自 MatMul 或 Conv2D ---
    const Edge* edge_in0;
    if (!bias_node->input_edge(0, &edge_in0).ok()) continue;
    Node* matmul_node = edge_in0->src();
    
    // --- 检查输入 1: Bias Tensor ---
    const Edge* edge_in_bias;
    if (!bias_node->input_edge(1, &edge_in_bias).ok()) continue;
    Node* bias_tensor_node = edge_in_bias->src();
    int bias_tensor_idx = edge_in_bias->src_output();

    bool is_matmul = (matmul_node->type_string() == "MatMul");
    bool is_conv = (matmul_node->type_string() == "Conv2D");

    if (!is_matmul && !is_conv) continue;

    // 如果 MatMul 的结果被多处引用，不能融合 BiasAdd
    if (CountConsumers(matmul_node) > 1) continue;

    // =========================================================
    // 向前多看一步，寻找 Relu
    // =========================================================
    Node* relu_node = nullptr;
    
    // 只有当 BiasAdd 只有一个消费者时，才尝试寻找 Relu
    // 防止 BiasAdd 的结果被分叉使用
    if (CountConsumers(bias_node) == 1) {
        for (const Edge* e : bias_node->out_edges()) {
            if (!e->IsControlEdge() && e->dst()->type_string() == "Relu") {
                relu_node = e->dst();
                break;
            }
        }
    }

    // 确定最终的输出源节点：
    // 如果融合了 Relu，下游消费者原本是连在 Relu 上的
    // 如果没融合 Relu，下游消费者原本是连在 BiasAdd 上的
    Node* final_output_source = (relu_node != nullptr) ? relu_node : bias_node;

    // 打印融合计划
    std::string fuse_msg = matmul_node->name() + " + " + bias_node->name();
    if (relu_node) fuse_msg += " + " + relu_node->name();
    
//    fprintf(stderr, "[MUSA_FUSE] Fusing %s -> %s\n", 
  //          fuse_msg.c_str(), 
    //        (is_conv ? "_FusedConv2D" : "MusaFusedMatMul"));

    // --- 准备 MatMul 的原始输入 ---
    const Edge* edge_mm_a; 
    if (!matmul_node->input_edge(0, &edge_mm_a).ok()) continue;
    const Edge* edge_mm_b; 
    if (!matmul_node->input_edge(1, &edge_mm_b).ok()) continue;
    
    Node* node_a = edge_mm_a->src(); int idx_a = edge_mm_a->src_output();
    Node* node_b = edge_mm_b->src(); int idx_b = edge_mm_b->src_output();

    // --- 收集下游消费者 (从 final_output_source 收集) ---
    std::vector<std::pair<Node*, int>> consumers;
    for (const Edge* e : final_output_source->out_edges()) {
      if (!e->IsControlEdge()) {
        consumers.push_back({e->dst(), e->dst_input()});
      }
    }

    // --- 创建新节点定义 ---
    NodeDef new_def;
    // 使用 BiasAdd 的名字作为基础，避免命名冲突
    new_def.set_name(bias_node->name()); 
    new_def.set_op(is_conv ? "_FusedConv2D" : "MusaFusedMatMul");
    new_def.set_device(bias_node->requested_device());
    
    auto* attr = new_def.mutable_attr();
    const auto& mm_attrs = matmul_node->attrs();
    
    // 复制 MatMul 属性
    if (mm_attrs.Find("T")) (*attr)["T"] = *mm_attrs.Find("T");
    if (mm_attrs.Find("transpose_a")) (*attr)["transpose_a"] = *mm_attrs.Find("transpose_a");
    if (mm_attrs.Find("transpose_b")) (*attr)["transpose_b"] = *mm_attrs.Find("transpose_b");
    if (is_conv) {
        if (mm_attrs.Find("strides")) (*attr)["strides"] = *mm_attrs.Find("strides");
        if (mm_attrs.Find("padding")) (*attr)["padding"] = *mm_attrs.Find("padding");
    }

    // --- 设置 fused_ops 列表 ---
    auto* fused_ops_list = (*attr)["fused_ops"].mutable_list();
    fused_ops_list->add_s("BiasAdd");
    if (relu_node) {
        fused_ops_list->add_s("Relu"); // 如果找到了 Relu，追加进去
    }

    (*attr)["num_args"].set_i(1);
    (*attr)["epsilon"].set_f(0.0001f);
    
    // --- 修改图结构 ---
    // 移除旧节点
    graph->RemoveNode(bias_node);
    graph->RemoveNode(matmul_node);
    if (relu_node) {
        graph->RemoveNode(relu_node);
    }
    
    // 添加新节点
    Status status;
    Node* new_node = graph->AddNode(new_def, &status);
    if (!status.ok()) {
        fprintf(stderr, "[MUSA_ERROR] Failed to add fused node: %s\n", status.error_message().c_str());
        continue;
    }
    
    // 重连输入边
    graph->AddEdge(node_a, idx_a, new_node, 0);
    graph->AddEdge(node_b, idx_b, new_node, 1);
    graph->AddEdge(bias_tensor_node, bias_tensor_idx, new_node, 2);
    
    // 重连输出边 (连接到原来 Relu 或 BiasAdd 的消费者)
    for (auto& c : consumers) {
        graph->AddEdge(new_node, 0, c.first, c.second);
    }
    
    graph_changed = true;
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 10, MusaOptimizationPass);

void ForceMusaOptimizationPassRegistration() {}

} // namespace tensorflow

