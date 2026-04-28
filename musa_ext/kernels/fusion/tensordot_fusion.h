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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_FUSION_H_

#include <string>
#include <vector>

#include "kernels/fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * TensorDot Fusion Pattern
 *
 * 精确匹配路径 (从终点到起点):
 *
 *   第一层(起始层): Shape_1, Transpose
 *   第二层: GatherV2_1(←Shape_1), GatherV2_2(←Shape_1)
 *   第三层: Prod_1(←GatherV2_1), Prod_2(←GatherV2_2), ConcatV2(←GatherV2_1)
 *   第四层: Pack(←Prod_1, Prod_2)
 *   第五层: Reshape_1(←Transpose, Pack)
 *   第六层: MatMul(←Reshape_1)
 *   第七层(终点层): Reshape_2(←MatMul, ConcatV2)
 *
 *
 * 融合后生成: MusaTensorDot op
 * 输入: 来自 Shape_1 和 Transpose 的输入
 */
class MusaTensorDotFusion : public FusionPattern {
 public:
  MusaTensorDotFusion();
  ~MusaTensorDotFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 100; }
  bool IsKernelAvailable() const override;
  std::string GetName() const override { return "MusaTensorDotFusion"; }
  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaTensorDot kernel not available on this device";
    }
    return "";
  }

 private:
  // 从终点 Reshape_2 开始向上匹配
  FusionMatchResult MatchFromReshapeNode(const GraphDef& graph,
                                         int reshape_node_idx) const;

  mutable bool kernel_checked_ = false;
  mutable bool kernel_available_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_TENSORDOT_FUSION_H_
