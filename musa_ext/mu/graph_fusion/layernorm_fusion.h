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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_LAYERNORM_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_LAYERNORM_FUSION_H_

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * LayerNorm Fusion Pattern
 *
 * Matches the pattern: AddV2(Mul(MusaNormalize, gamma), beta)
 *
 *   Layer 1 (start): MusaNormalize - Normalize input (already fused from
 * normalize pattern) Layer 2:         Mul           - Scale by gamma Layer 3
 * (end):   AddV2         - Add beta bias
 *
 * Inputs:
 *   - x: Original input tensor (from MusaNormalize's first input)
 *   - gamma: Scale parameter (Const or ExpandDims of Const)
 *   - beta: Bias parameter (Const or ExpandDims of Const)
 *   - epsilon: From MusaNormalize's epsilon attribute (default 1e-5)
 *
 * Output: LayerNorm(x, gamma, beta)
 *
 * Fused op: MusaLayerNorm
 */
class MusaLayerNormFusion : public FusionPattern {
 public:
  MusaLayerNormFusion();
  ~MusaLayerNormFusion() override = default;

  // Match the LayerNorm pattern starting from AddV2 node (layer 11)
  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  // Apply the fusion: replace matched subgraph with MusaLayerNorm
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  // Priority: higher than basic patterns
  int GetPriority() const override { return 1; }

  // Kernel is available (implemented in musa_layernorm_op.cc)
  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "MusaLayerNormFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaLayerNorm kernel not available on this device";
    }
    return "";
  }

 private:
  // Match LayerNorm pattern starting from AddV2 node
  // Pattern: AddV2(Mul(MusaNormalize, gamma), beta)
  FusionMatchResult MatchFromAddNode(const GraphDef& graph,
                                     int add_node_idx) const;

  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_LAYERNORM_FUSION_H_