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

// LayerNorm fusion pattern
// Matches a subgraph that implements LayerNorm and replaces it with MusaLayerNorm op
//
// Pattern structure (typical TF implementation):
//   input -> Mean -> Sub -> SquaredDifference -> Mean -> Add(epsilon) -> Rsqrt -> Mul
//                                                                    -> Mul -> Add (with gamma/beta)
//
// Or the variable-based version:
//   input -> Moments -> [mean, variance] -> Sub/Sqrt/Div/Mul/Add pattern
//
// The fused version:
//   input -> MusaLayerNorm -> output
//   (with gamma and beta as inputs)

class MusaLayerNormFusion : public FusionPattern {
 public:
  MusaLayerNormFusion();
  ~MusaLayerNormFusion() override = default;
  
  // Match the LayerNorm pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const override;
  
  // Apply the fusion: replace matched subgraph with MusaLayerNorm
  Status Apply(GraphDef* graph, const FusionMatchResult& match_result) const override;
  
  // Priority: higher than basic patterns
  int GetPriority() const override { return 100; }
  
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
  // Match LayerNorm pattern starting from Add node (beta addition)
  // This is the most reliable matching strategy
  FusionMatchResult MatchFromAddNode(const GraphDef& graph, int add_node_idx) const;
  
  // Match LayerNorm pattern starting from RealDiv node
  // Alternative matching strategy for different graph structures
  FusionMatchResult MatchFromDivNode(const GraphDef& graph, int div_node_idx) const;
  
  // Kernel availability flag
  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_LAYERNORM_FUSION_H_
