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

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * MusaShiftedAffineMap Fusion Pattern
 *
 * Target fused op:
 *   MusaShiftedAffineMap(input, scale_slice, const_term, bias_slice)
 *
 * Match entry:
 *   start from the downstream Select/SelectV2 node and trace the chosen
 *   branch upward.
 *
 * Expected arithmetic core:
 *   output = AddV2(Mul(input, AddV2(scale_slice, const_term)), bias_slice)
 *
 * Where both scale_slice and bias_slice are required to be produced by a
 * StridedSlice chain validated during matching.
 */
class MusaShiftedAffineMapFusion : public FusionPattern {
 public:
  MusaShiftedAffineMapFusion();
  ~MusaShiftedAffineMapFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 105; }
  bool IsKernelAvailable() const override;
  std::string GetName() const override {
    return "MusaShiftedAffineMapFusion";
  }
  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaShiftedAffineMap kernel not available on this device";
    }
    return "";
  }

 private:
  FusionMatchResult MatchFromSelectNode(const GraphDef& graph,
                                        int select_node_idx) const;

  mutable bool kernel_checked_ = false;
  mutable bool kernel_available_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_SHIFTED_AFFINE_MAP_FUSION_H_
