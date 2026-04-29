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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_SHIFTED_AFFINE_MAP_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_SHIFTED_AFFINE_MAP_FUSION_H_

#include <string>
#include <vector>

#include "kernels/fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * ShiftedAffineMap Fusion Pattern
 *
 * Matches the pattern (Top-Down, Post-Constant-Folding):
 * AddV2 (output_add)
 * ├─ Mul
 * │   ├─ Const (const_left)
 * │   └─ Select (mask)
 * └─ Const (const_right)
 *
 * Semantics:
 * output = mask * const_left + const_right
 */

class MusaShiftedAffineMapFusion : public FusionPattern {
 public:
  MusaShiftedAffineMapFusion();
  ~MusaShiftedAffineMapFusion() override = default;

  // Match the pattern starting from a node
  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  // Apply the fusion: replace matched subgraph with MusaShiftedAffineMap
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  // Priority: adjust as needed relative to other patterns
  int GetPriority() const override { return 90; }

  // Check if the corresponding kernel is available
  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "MusaShiftedAffineMapFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaShiftedAffineMap kernel not available on this device";
    }
    return "";
  }

 private:
  // Match starting from the top-level AddV2 output node.
  FusionMatchResult MatchFromOutputAddNode(const GraphDef& graph,
                                           int add_node_idx) const;

  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_SHIFTED_AFFINE_MAP_FUSION_H_
