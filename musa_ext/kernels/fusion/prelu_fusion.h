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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PRELU_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PRELU_FUSION_H_

#include <string>
#include <vector>

#include "kernels/fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// PRelu fusion pattern
// Matches a subgraph that implements PRelu and replaces it with MusaPRelu op
//
// PRelu semantics: f(x) = max(0, x) + alpha * min(0, x)
//
// Pattern structure (matched from AddV2 backward):
//
//   AddV2 (output)
//   ├── input[0]: Relu1
//   │   └── input: Select (x, cond, then, else)
//   │
//   └── input[1]: Mul
//       ├── input[0]: Neg1
//       │   └── input: Identity/Const (alpha parameter)
//       │
//       └── input[1]: Relu2
//           └── input: Neg2
//               └── input: Select (same as Relu1's input)
//
// The fused version:
//   x ──► MusaPRelu ◄── alpha
//          │
//          ▼
//       (output)
//
// Note: The output (AddV2) may have downstream consumers which will be
// reconnected to the fused node's output.

class MusaPReluFusion : public FusionPattern {
 public:
  MusaPReluFusion();
  ~MusaPReluFusion() override = default;

  // Match the PRelu pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  // Apply the fusion: replace matched subgraph with MusaPRelu
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  // Priority: standard priority for activation fusion patterns
  int GetPriority() const override { return 60; }

  // Kernel is available (implemented in musa_prelu_op.cc)
  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "MusaPReluFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaPRelu kernel not available on this device";
    }
    return "";
  }

 private:
  // Match PRelu pattern starting from AddV2 node
  FusionMatchResult MatchFromAddV2Node(const GraphDef& graph,
                                       int addv2_node_idx) const;

  // Kernel availability flag
  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_PRELU_FUSION_H_
