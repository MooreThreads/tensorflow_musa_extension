/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace {

// Device type for MUSA
constexpr char kMusaDeviceType[] = "MUSA";

// Tri-state configuration for optimizers
enum class TriState { kDefault = 0, kOff = 1, kOn = 2 };

// Optimizer configurations - controls interaction with TensorFlow built-in
// optimizers Based on TF Modular Graph C API TP_OptimizerConfigs
struct MusaOptimizerConfigs {
  TriState disable_model_pruning = TriState::kDefault;
  TriState implementation_selector = TriState::kDefault;
  TriState function_optimization = TriState::kDefault;
  TriState common_subgraph_elimination = TriState::kDefault;
  TriState arithmetic_optimization = TriState::kDefault;
  TriState debug_stripper = TriState::kDefault;
  TriState constant_folding = TriState::kDefault;
  TriState shape_optimization = TriState::kDefault;
  TriState auto_mixed_precision =
      TriState::kOff;  // MUSA handles AMP internally
  TriState pin_to_host_optimization = TriState::kDefault;
  TriState layout_optimizer = TriState::kOff;  // MUSA handles layout internally
  TriState remapping = TriState::kDefault;
  TriState loop_optimization = TriState::kDefault;
  TriState dependency_optimization = TriState::kDefault;
  TriState memory_optimization = TriState::kDefault;
  TriState auto_parallel = TriState::kDefault;
  TriState scoped_allocator_optimization = TriState::kDefault;
};

// MUSA AMP Configuration
class MusaAmpConfig {
 public:
  std::unordered_set<string> fp16_compute_ops = {"MatMul",
                                                 "BatchMatMul",
                                                 "BatchMatMulV2",
                                                 "Conv2D",
                                                 "Conv2DBackpropInput",
                                                 "Conv2DBackpropFilter",
                                                 "DepthwiseConv2dNative",
                                                 "Conv3D",
                                                 "FusedBatchNorm",
                                                 "FusedBatchNormV2",
                                                 "FusedBatchNormV3"};

  std::unordered_set<string> fp32_keep_ops = {
      "Softmax",
      "LogSoftmax",
      "SoftmaxCrossEntropyWithLogits",
      "SparseSoftmaxCrossEntropyWithLogits",
      "SigmoidCrossEntropyWithLogits",
      "Mean",
      "Sum",
      "Prod",
      "L2Loss",
      "Norm",
      "Exp",
      "Log",
      "Sqrt",
      "Rsqrt",
      "Reciprocal",
      "Square"};

  std::unordered_set<string> conditional_ops = {
      "Add", "AddV2", "Sub", "Mul", "Div", "BiasAdd", "BiasAddGrad"};

  std::unordered_set<string> activation_ops = {
      "Relu", "Relu6", "Elu", "Selu", "LeakyRelu", "Sigmoid", "Tanh"};

  bool aggressive_mode = false;
  DataType target_dtype = DT_HALF;
};

// Graph utilities
class MusaGraphUtils {
 public:
  static NodeDef* CreateConstNode(GraphDef* graph, const string& name,
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

  static NodeDef* InsertTranspose(GraphDef* graph, const string& base_name,
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

  static NodeDef* InsertCast(GraphDef* graph, const string& name,
                             const string& input_name, DataType src_dtype,
                             DataType dst_dtype, const string& device) {
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

  static void RedirectEdges(GraphDef* graph, const string& old_node_name,
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

  static void RewriteLayoutAttributes(NodeDef* node) {
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

  static bool IsMusaNCHWSupported(const NodeDef& node) {
    if (node.device().find(kMusaDeviceType) == std::string::npos) return false;
    return kLayoutSensitiveOps(node) || kLayoutAgnosticOps(node);
  }

  static bool kLayoutSensitiveOps(const NodeDef& node) {
    static const std::unordered_set<string> sensitive_ops = {
        "Conv2D",  "DepthwiseConv2dNative", "MaxPool",
        "AvgPool", "FusedBatchNorm",        "FusedBatchNormV3"};
    return sensitive_ops.count(node.op()) > 0;
  }

  static bool kLayoutAgnosticOps(const NodeDef& node) {
    static const std::unordered_set<string> agnostic_ops = {
        "Relu", "Sigmoid", "Tanh", "BiasAdd", "Add", "Sub", "Mul", "Identity"};
    return agnostic_ops.count(node.op()) > 0;
  }
};

// Check if graph contains MUSA device nodes
bool GraphHasMusaNodes(const GraphDef& graph) {
  for (const auto& node : graph.node()) {
    if (node.device().find(kMusaDeviceType) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

// Unified MUSA Graph Optimizer
// Combines Layout optimization and AMP (Automatic Mixed Precision)
// Based on Modular TensorFlow Graph C API design principles
class MusaGraphOptimizer : public CustomGraphOptimizer {
 public:
  MusaGraphOptimizer() : device_type_(kMusaDeviceType) {}
  ~MusaGraphOptimizer() override {}

  std::string name() const override { return "musa_graph_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    if (config) {
      for (const auto& param : config->parameter_map()) {
        if (param.first == "aggressive_mode") {
          amp_config_.aggressive_mode = param.second.b();
        } else if (param.first == "precision_mode") {
          string mode = param.second.s();
          if (mode == "BF16" || mode == "BFLOAT16") {
            amp_config_.target_dtype = DT_BFLOAT16;
          } else {
            amp_config_.target_dtype = DT_HALF;
          }
        } else if (param.first == "disable_layout_optimizer") {
          // Allow user to disable layout optimization
          if (param.second.b()) {
            configs_.layout_optimizer = TriState::kOff;
          }
        } else if (param.first == "disable_amp") {
          // Allow user to disable AMP
          if (param.second.b()) {
            configs_.auto_mixed_precision = TriState::kOff;
          }
        }
      }
    }
    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;

    // Skip optimization if graph doesn't contain MUSA nodes
    if (!GraphHasMusaNodes(*optimized_graph)) {
      VLOG(2)
          << "MusaGraphOptimizer: No MUSA nodes found, skipping optimization";
      return Status::OK();
    }

    VLOG(1) << "MusaGraphOptimizer: Optimizing graph with "
            << optimized_graph->node_size() << " nodes";

    // Step 1: Layout optimization (NHWC -> NCHW)
    if (configs_.layout_optimizer != TriState::kOff) {
      OptimizeLayout(optimized_graph);
    }

    // Step 2: AMP optimization (FP32 -> FP16)
    if (configs_.auto_mixed_precision != TriState::kOff) {
      OptimizeAMP(optimized_graph);
    }

    VLOG(1) << "MusaGraphOptimizer: Optimization complete, graph now has "
            << optimized_graph->node_size() << " nodes";

    return Status::OK();
  }

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

  // Get optimizer configurations - used for coordination with other optimizers
  const MusaOptimizerConfigs& GetConfigs() const { return configs_; }

 private:
  MusaAmpConfig amp_config_;
  MusaOptimizerConfigs configs_;
  string device_type_;

  // Layout Optimization
  void OptimizeLayout(GraphDef* graph) {
    bool changed = true;
    int iteration = 0;
    const int kMaxIterations = 5;

    while (changed && iteration < kMaxIterations) {
      changed = false;
      iteration++;

      for (int i = 0; i < graph->node_size(); ++i) {
        NodeDef* node = graph->mutable_node(i);

        if (!MusaGraphUtils::IsMusaNCHWSupported(*node)) {
          continue;
        }

        auto* attr = node->mutable_attr();
        bool is_already_nchw = (attr->count("data_format") &&
                                (*attr)["data_format"].s() == "NCHW");
        if (is_already_nchw) continue;

        bool has_nchw_upstream = false;
        if (node->input_size() > 0) {
          if (node->input(0).find("/post_transpose_nhwc") !=
              std::string::npos) {
            has_nchw_upstream = true;
          }
        }

        bool should_transform = false;
        if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
          should_transform = true;
        } else if (MusaGraphUtils::kLayoutAgnosticOps(*node) &&
                   has_nchw_upstream) {
          should_transform = true;
        }

        if (should_transform) {
          std::string op_name = node->name();
          DataType dtype = (*attr)["T"].type();
          std::string device = node->device();

          if (has_nchw_upstream) {
            std::string real_src = node->input(0).substr(
                0, node->input(0).find("/post_transpose_nhwc"));
            node->set_input(0, real_src);
          } else {
            std::string pre_name = op_name + "/pre_transpose_nchw";
            MusaGraphUtils::InsertTranspose(graph, pre_name, node->input(0),
                                            {0, 3, 1, 2}, dtype, device);
            node->set_input(0, pre_name);
          }

          (*attr)["data_format"].set_s("NCHW");
          if (MusaGraphUtils::kLayoutSensitiveOps(*node)) {
            MusaGraphUtils::RewriteLayoutAttributes(node);
          }

          std::string post_name = op_name + "/post_transpose_nhwc";
          MusaGraphUtils::InsertTranspose(graph, post_name, op_name,
                                          {0, 2, 3, 1}, dtype, device);
          MusaGraphUtils::RedirectEdges(graph, op_name, post_name);

          changed = true;
        }
      }
    }
  }

  // AMP Optimization
  void OptimizeAMP(GraphDef* graph) {
    std::unordered_map<string, bool> should_convert;
    AnalyzeGraphForAMP(*graph, should_convert);

    int original_node_size = graph->node_size();
    for (int i = 0; i < original_node_size; ++i) {
      NodeDef* node = graph->mutable_node(i);

      if (node->device().find(kMusaDeviceType) == std::string::npos) {
        continue;
      }

      if (!should_convert[node->name()]) {
        continue;
      }

      DataType dtype = GetNodeDataType(node);
      if (dtype != DT_FLOAT) {
        continue;
      }

      ConvertNodeToLowPrecision(graph, node);
    }
  }

  void AnalyzeGraphForAMP(const GraphDef& graph,
                          std::unordered_map<string, bool>& should_convert) {
    std::unordered_map<string, const NodeDef*> node_map;
    for (const auto& node : graph.node()) {
      node_map[node.name()] = &node;
    }

    for (const auto& node : graph.node()) {
      bool convert = false;

      if (amp_config_.fp16_compute_ops.count(node.op())) {
        convert = true;
      }

      if (amp_config_.fp32_keep_ops.count(node.op())) {
        convert = false;
      }

      if (amp_config_.activation_ops.count(node.op())) {
        if (node.input_size() > 0) {
          string input_name = GetNodeNameFromInput(node.input(0));
          if (node_map.count(input_name)) {
            const NodeDef* input_node = node_map.at(input_name);
            if (amp_config_.fp16_compute_ops.count(input_node->op())) {
              convert = true;
            }
          }
        }
      }

      if (amp_config_.conditional_ops.count(node.op())) {
        if (amp_config_.aggressive_mode) {
          convert = true;
        } else {
          int low_prec_inputs = 0;
          for (const auto& input : node.input()) {
            if (input[0] == '^') continue;
            string input_name = GetNodeNameFromInput(input);
            if (node_map.count(input_name)) {
              const NodeDef* input_node = node_map.at(input_name);
              if (amp_config_.fp16_compute_ops.count(input_node->op())) {
                low_prec_inputs++;
              }
            }
          }
          if (low_prec_inputs >= 1) {
            convert = true;
          }
        }
      }

      should_convert[node.name()] = convert;
    }
  }

  string GetNodeNameFromInput(const string& input) {
    if (input.empty()) return "";
    if (input[0] == '^') return input.substr(1);

    size_t colon_pos = input.find(':');
    if (colon_pos != std::string::npos) {
      return input.substr(0, colon_pos);
    }
    return input;
  }

  DataType GetNodeDataType(const NodeDef* node) {
    if (node->attr().count("T")) {
      return node->attr().at("T").type();
    } else if (node->attr().count("dtype")) {
      return node->attr().at("dtype").type();
    }
    return DT_INVALID;
  }

  bool ConvertNodeToLowPrecision(GraphDef* graph, NodeDef* node) {
    string op_name = node->name();
    string device = node->device();
    DataType target_t = amp_config_.target_dtype;

    if (node->mutable_attr()->count("T")) {
      (*node->mutable_attr())["T"].set_type(target_t);
    } else if (node->mutable_attr()->count("dtype")) {
      (*node->mutable_attr())["dtype"].set_type(target_t);
    }

    std::vector<string> new_inputs;
    for (int idx = 0; idx < node->input_size(); ++idx) {
      string input_name = node->input(idx);

      if (input_name.empty() || input_name[0] == '^') {
        new_inputs.push_back(input_name);
        continue;
      }

      if (input_name.find("/CastF2Lower") != std::string::npos) {
        new_inputs.push_back(input_name);
        continue;
      }

      string cast_in_name =
          op_name + "/Input_" + std::to_string(idx) + "/CastF2Lower";

      MusaGraphUtils::InsertCast(graph, cast_in_name, input_name, DT_FLOAT,
                                 target_t, device);
      new_inputs.push_back(cast_in_name);
    }

    node->clear_input();
    for (const auto& input : new_inputs) {
      node->add_input(input);
    }

    string cast_out_name = op_name + "/Output/CastLower2F";
    MusaGraphUtils::InsertCast(graph, cast_out_name, op_name, target_t,
                               DT_FLOAT, device);

    for (int j = 0; j < graph->node_size(); ++j) {
      NodeDef* consumer = graph->mutable_node(j);
      if (consumer->name() == cast_out_name) continue;

      for (int k = 0; k < consumer->input_size(); ++k) {
        string inp = consumer->input(k);

        if (inp == op_name) {
          consumer->set_input(k, cast_out_name);
        } else if (inp.find(op_name + ":") == 0) {
          string suffix = inp.substr(op_name.length());
          consumer->set_input(k, cast_out_name + suffix);
        } else if (inp == "^" + op_name) {
          consumer->set_input(k, "^" + cast_out_name);
        }
      }
    }

    return true;
  }
};

REGISTER_GRAPH_OPTIMIZER_AS(MusaGraphOptimizer, "musa_graph_optimizer");

}  // namespace grappler
}  // namespace tensorflow

extern "C" {
// This function will be called when the plugin is loaded
// Note: Full C API (TF_InitGraphPlugin) is available in TensorFlow 2.5+
// For TensorFlow 2.4.4, we use C++ API with REGISTER_GRAPH_OPTIMIZER_AS
void __attribute__((constructor)) ForceMusaGraphOptimizerLoad() {
  // Optimizer is automatically registered via REGISTER_GRAPH_OPTIMIZER_AS
  VLOG(1) << "MUSA Graph Optimizer plugin loaded (v2.4.4 C++ API mode)";
}
}
