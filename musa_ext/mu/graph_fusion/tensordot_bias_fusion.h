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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_BIAS_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_BIAS_FUSION_H_

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * TensorDot + BiasAdd Fusion Pattern
 *
 * Matches the pattern:
 *   BiasAdd
 *     ├─ TensorDot 子图输出 (Reshape_2)
 *     └─ bias (权重节点)
 *
 * The TensorDot subgraph is the same as in MusaTensorDotFusion:
 *   Shape_1, Transpose
 *     ↓
 *   GatherV2_1, GatherV2_2
 *     ↓
 *   Prod_1, Prod_2
 *     ↓
 *   Pack
 *     ↓
 *   Reshape_1
 *     ↓
 *   MatMul (with weight)
 *     ↓
 *   ConcatV2
 *     ↓
 *   Reshape_2
 *
 * Fused output: MusaTensorDotBias op
 */
class MusaTensorDotBiasFusion : public FusionPattern {
 public:
  MusaTensorDotBiasFusion();
  ~MusaTensorDotBiasFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                        int start_node_idx) const override;
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 105; }  // Slightly higher than MusaTensorDotFusion
  bool IsKernelAvailable() const override;
  std::string GetName() const override { return "MusaTensorDotBiasFusion"; }
  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaTensorDotBias kernel not available on this device";
    }
    return "";
  }

 private:
  // Match from BiasAdd node (entry point)
  FusionMatchResult MatchFromBiasAddNode(const GraphDef& graph,
                                        int bias_add_node_idx) const;

  mutable bool kernel_checked_ = false;
  mutable bool kernel_available_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_BIAS_FUSION_H_
