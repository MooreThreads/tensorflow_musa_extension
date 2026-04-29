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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSELAYERNORMV2_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSELAYERNORMV2_FUSION_H_

#include <string>

#include "kernels/fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// FuseLayerNormV2 fusion pattern.
//
// Matches the following pattern:
//   x -> Shape -> Slice(d0,d1,d2)
//   x -> Reshape([1, d0*d1, d2, 1])
//     -> FusedBatchNormV3(scale=Fill([d0*d1],1), offset=Fill([d0*d1],0),
//                         data_format=NCHW, is_training=true, epsilon=eps)
//     -> Reshape(Shape(x))
//     -> Mul(gamma[d2]) -> Add(beta[d2])
//
// Replaces it with:
//   MusaLayerNorm(x, gamma, beta, epsilon=eps)
class MusaFuseLayerNormV2Fusion : public FusionPattern {
 public:
  MusaFuseLayerNormV2Fusion();
  ~MusaFuseLayerNormV2Fusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 110; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "MusaFuseLayerNormV2Fusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaLayerNorm kernel not available on this device";
    }
    return "";
  }

 private:
  FusionMatchResult MatchFromAddNode(const GraphDef& graph,
                                     int add_node_idx) const;

  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_FUSELAYERNORMV2_FUSION_H_
