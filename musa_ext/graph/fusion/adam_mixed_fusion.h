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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_ADAM_MIXED_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_ADAM_MIXED_FUSION_H_

#include <string>

#include "graph/fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Rewrites
//   bf16/fp16 grad -> Cast(DstT=fp32) -> ResourceApplyAdam(T=fp32, ...)
// into
//   bf16/fp16 grad -> MusaResourceApplyAdamMixed(T=bf16/fp16, ...)
//
// State variables (var/m/v) stay fp32. The Cast disappears and grad is loaded
// in its low-precision form directly inside the new op, saving one gradient
// sized memory pass per step. Numerics are unchanged: both paths promote the
// gradient to fp32 before doing the Adam math; we just skip materializing the
// intermediate fp32 gradient tensor.
//
// The rewrite is conservative: it only fires when the matched Cast has a
// single ResourceApplyAdam consumer and the Adam node itself has T=DT_FLOAT
// (i.e. the caller already went through Keras-style mixed_bfloat16 plumbing).
// Disable via env var MUSA_DISABLE_ADAM_MIXED_FUSION=1 or by adding
// "MusaAdamMixedFusion" to the disabled fusion patterns list.
class MusaAdamMixedFusion : public FusionPattern {
 public:
  MusaAdamMixedFusion();
  ~MusaAdamMixedFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  // Lower priority than the big op-shape fusions (Gelu/LayerNorm/Normalize)
  // because Adam doesn't compose with them; the relative order only matters
  // when patterns share matched nodes, which is not the case here.
  int GetPriority() const override { return 40; }
  bool IsKernelAvailable() const override;
  std::string GetName() const override { return "MusaAdamMixedFusion"; }

 private:
  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_ADAM_MIXED_FUSION_H_
