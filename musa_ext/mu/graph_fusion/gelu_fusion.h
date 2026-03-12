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

// GELU fusion pattern.
// The exact-erf path is the primary target in current large-model graphs.
// The tanh-approximate path is kept in a separate matcher so it can be
// reasoned about and debugged independently.

class MusaGeluFusion : public FusionPattern {
 public:
  MusaGeluFusion();
  ~MusaGeluFusion() override = default;
  
  // Match the GELU pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph, int start_node_idx) const override;
  
  // Apply the fusion by replacing the matched output with MusaGelu.
  Status Apply(GraphDef* graph, const FusionMatchResult& match_result) const override;
  
  // Priority: same as other activation fusions
  int GetPriority() const override { return 90; }
  
  bool IsKernelAvailable() const override;
  
  std::string GetName() const override { return "MusaGeluFusion"; }
  
  std::string GetFallbackReason() const override { return ""; }

 private:
  // Match exact GELU patterns used by TensorFlow's erf-based formulation:
  //   0.5 * x * (1 + erf(x / sqrt(2)))
  //   0.5 * x * erfc(-x / sqrt(2)))
  FusionMatchResult MatchStandardPattern(const GraphDef& graph, int start_node_idx) const;
  
  // Match the optional tanh-approximate GELU path:
  //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  FusionMatchResult MatchApproximatePattern(const GraphDef& graph, int start_node_idx) const;

  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_GELU_FUSION_H_
