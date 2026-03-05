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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_GELU_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_GELU_FUSION_H_

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// GELU fusion pattern
// Matches a subgraph that implements GELU activation and would replace it with MusaGelu op
//
// GELU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Pattern structure (TF implementation):
//   input -> Div(sqrt(2)) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5)
//   input --------------------------------------->
//
// The fused version (NOT YET IMPLEMENTED - for fallback testing):
//   input -> MusaGelu -> output
//
// This pattern is used to test the fallback mechanism when kernel is not available

class MusaGeluFusion : public FusionPattern {
 public:
  MusaGeluFusion();
  ~MusaGeluFusion() override = default;
  
  // Match the GELU pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const override;
  
  // Apply the fusion: would replace matched subgraph with MusaGelu
  // NOTE: Currently this returns OK but does NOT apply the fusion since kernel
  // is not implemented. This is intentional for fallback mechanism testing.
  Status Apply(GraphDef* graph, const FusionMatchResult& match_result) const override;
  
  // Priority: same as other activation fusions
  int GetPriority() const override { return 90; }
  
  // Kernel is NOT available (gelu not yet implemented)
  // This is intentional for testing the fallback mechanism
  bool IsKernelAvailable() const override;
  
  std::string GetName() const override { return "MusaGeluFusion"; }
  
  std::string GetFallbackReason() const override {
    return "MusaGelu kernel not yet implemented - using fallback to standard ops";
  }

 private:
  // Match standard GELU pattern
  FusionMatchResult MatchStandardPattern(const GraphDef& graph, int start_node_idx) const;
  
  // Match approximate GELU pattern (using tanh approximation)
  FusionMatchResult MatchApproximatePattern(const GraphDef& graph, int start_node_idx) const;
  
  // Helper: Check if a node is part of sqrt(2) computation
  bool IsSqrt2Div(const NodeDef& node, const GraphDef& graph) const;
  
  mutable bool kernel_available_ = false;  // Not implemented yet
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_GELU_FUSION_H_
