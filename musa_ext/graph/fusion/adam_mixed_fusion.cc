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

#include "graph/fusion/adam_mixed_fusion.h"

#include <cstdlib>
#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsTruthyEnvVar(const char* env_name) {
  const char* env_val = std::getenv(env_name);
  if (!env_val) return false;
  const std::string value(env_val);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
         value == "YES" || value == "on" || value == "ON";
}

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  const std::string producer_name =
      FusionGraphUtils::GetProducerNodeName(input);
  if (producer_name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, producer_name);
}

DataType GetAttrType(const NodeDef& node, const std::string& key) {
  const auto it = node.attr().find(key);
  if (it == node.attr().end()) return DT_INVALID;
  return it->second.type();
}

bool IsMusaDevice(const NodeDef& node) {
  return node.device().find("MUSA") != std::string::npos;
}

// Number of consumers of `producer_name` excluding the Adam node itself.
// Used to decide whether the Cast can be safely deleted after the rewrite.
int CountOtherConsumers(const GraphDef& graph, const std::string& producer_name,
                        const std::string& adam_name) {
  int count = 0;
  for (const auto& node : graph.node()) {
    if (node.name() == adam_name || node.name() == producer_name) continue;
    for (int i = 0; i < node.input_size(); ++i) {
      if (FusionGraphUtils::GetProducerNodeName(node.input(i)) ==
          producer_name) {
        ++count;
        break;
      }
    }
  }
  return count;
}

}  // namespace

MusaAdamMixedFusion::MusaAdamMixedFusion() = default;

bool MusaAdamMixedFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = !IsTruthyEnvVar("MUSA_DISABLE_ADAM_MIXED_FUSION");
    kernel_checked_ = true;
    if (kernel_available_) {
      VLOG(1) << "MusaAdamMixedFusion enabled";
    } else {
      VLOG(1)
          << "MusaAdamMixedFusion disabled via MUSA_DISABLE_ADAM_MIXED_FUSION";
    }
  }
  return kernel_available_;
}

FusionMatchResult MusaAdamMixedFusion::Match(const GraphDef& graph,
                                             int start_node_idx) const {
  FusionMatchResult result;
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return result;
  }

  const NodeDef& start = graph.node(start_node_idx);

  // We rewrite the stock ResourceApplyAdam (the op Keras Adam emits). We
  // intentionally do NOT rewrite our own MusaResourceApplyAdam<bfloat16> here
  // because that path is reached by users who explicitly keep state in bf16;
  // they need a different mitigation than a fusion rewrite.
  if (start.op() != "ResourceApplyAdam") return result;
  if (!IsMusaDevice(start)) return result;
  if (GetAttrType(start, "T") != DT_FLOAT) return result;

  // ResourceApplyAdam has 10 inputs: var, m, v, beta1_power, beta2_power, lr,
  // beta1, beta2, epsilon, grad. grad is input(9).
  if (start.input_size() < 10) return result;

  const NodeDef* grad_producer = FindProducer(graph, start.input(9));
  if (!grad_producer || grad_producer->op() != "Cast") return result;
  if (GetAttrType(*grad_producer, "DstT") != DT_FLOAT) return result;

  const DataType src_dtype = GetAttrType(*grad_producer, "SrcT");
  if (src_dtype != DT_BFLOAT16 && src_dtype != DT_HALF) return result;

  if (grad_producer->input_size() < 1) return result;

  result.matched = true;
  result.matched_nodes.push_back(&start);
  result.matched_nodes.push_back(grad_producer);
  result.captured_nodes["adam"] = &start;
  result.captured_nodes["cast"] = grad_producer;
  result.captured_attrs["grad_dtype"] =
      (src_dtype == DT_BFLOAT16) ? "BFLOAT16" : "HALF";
  return result;
}

Status MusaAdamMixedFusion::Apply(GraphDef* graph,
                                  const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return errors::InvalidArgument("Invalid Adam-mixed match");
  }
  if (!IsKernelAvailable()) return OkStatus();

  auto adam_it = match_result.captured_nodes.find("adam");
  auto cast_it = match_result.captured_nodes.find("cast");
  if (adam_it == match_result.captured_nodes.end() ||
      cast_it == match_result.captured_nodes.end() || !adam_it->second ||
      !cast_it->second) {
    return errors::InvalidArgument(
        "Missing captured nodes in Adam-mixed pattern");
  }

  const std::string adam_name = adam_it->second->name();
  const std::string cast_name = cast_it->second->name();

  // Resolve current graph indices (matched_nodes pointers are stale relative
  // to mutations performed by earlier Apply passes in the same sweep).
  int adam_idx = FusionGraphUtils::FindNodeIndex(*graph, adam_name);
  if (adam_idx < 0) {
    return errors::NotFound("Adam node disappeared before rewrite");
  }
  NodeDef* adam_node = graph->mutable_node(adam_idx);

  if (adam_node->op() == "MusaResourceApplyAdamMixed") {
    // Already rewritten on a prior sweep.
    return OkStatus();
  }
  if (adam_node->op() != "ResourceApplyAdam") {
    return OkStatus();
  }
  if (adam_node->input_size() < 10) {
    return OkStatus();
  }

  // Look up the Cast freshly; if a previous Apply already collapsed it we
  // bail out.
  const int cast_idx = FusionGraphUtils::FindNodeIndex(*graph, cast_name);
  if (cast_idx < 0) {
    return OkStatus();
  }
  const NodeDef& cast_node = graph->node(cast_idx);
  if (cast_node.op() != "Cast" || cast_node.input_size() < 1) {
    return OkStatus();
  }

  // Re-check the dtypes in case a previous rewrite changed them.
  if (GetAttrType(cast_node, "DstT") != DT_FLOAT) return OkStatus();
  const DataType src_dtype = GetAttrType(cast_node, "SrcT");
  if (src_dtype != DT_BFLOAT16 && src_dtype != DT_HALF) return OkStatus();

  const std::string original_grad_input = cast_node.input(0);
  if (original_grad_input.empty()) return OkStatus();

  // Count consumers BEFORE rewriting so we know whether the Cast can be
  // dropped. After rewriting, the Adam node no longer consumes it; if no one
  // else does either, RemoveNodesIfUnused will clean it up.
  const int other_consumers = CountOtherConsumers(*graph, cast_name, adam_name);

  // ------ Rewrite the Adam node ------
  adam_node->set_op("MusaResourceApplyAdamMixed");
  (*adam_node->mutable_attr())["T"].set_type(src_dtype);
  adam_node->set_input(9, original_grad_input);

  // ------ Garbage-collect the Cast if it's now orphaned ------
  int removed = 0;
  if (other_consumers == 0) {
    removed = FusionGraphUtils::RemoveNodesIfUnused(
        graph, {cast_name},
        /*protected_node_names=*/
        {adam_name,
         FusionGraphUtils::GetProducerNodeName(original_grad_input)});
  }

  VLOG(1) << "MusaAdamMixedFusion: rewrote " << adam_name
          << " (grad_dtype=" << DataTypeString(src_dtype)
          << ", cast_consumers_left=" << other_consumers
          << ", removed=" << removed << ")";

  return OkStatus();
}

REGISTER_FUSION_PATTERN(MusaAdamMixedFusion);
REGISTER_FUSION_KERNEL(MusaAdamMixedFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
