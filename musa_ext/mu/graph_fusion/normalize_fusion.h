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

#ifndef TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_NORMALIZE_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_NORMALIZE_FUSION_H_

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

/**
 * Normalize Fusion Pattern
 *
 * 匹配路径 (从起点到终点):
 *
 *   第一层(起始层): Mean_1 (计算均值)
 *   第二层: ExpandDims_1
 *   第三层: Sub (输入 - 均值)
 *   第四层: Square (平方)
 *   第五层: Mean_2 (计算方差)
 *   第六层: ExpandDims_2
 *   第七层: Sqrt (标准差)
 *   第八层: MusaClip (数值稳定性)
 *   第九层(终点层): RealDiv (归一化)
 *
 * 输入: 原始输入tensor (如 BiasAdd) + reduction_indices
 * 输出: 归一化后的tensor
 *
 * 融合后生成: MusaNormalize op
 */
class MusaNormalizeFusion : public FusionPattern {
 public:
  MusaNormalizeFusion();
  ~MusaNormalizeFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;
  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 105; }
  bool IsKernelAvailable() const override;
  std::string GetName() const override { return "MusaNormalizeFusion"; }
  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "MusaNormalize kernel not available on this device";
    }
    return "";
  }

 private:
  // 从终点 RealDiv 开始向上匹配
  FusionMatchResult MatchFromRealDivNode(const GraphDef& graph,
                                         int realdiv_node_idx) const;

  mutable bool kernel_checked_ = false;
  mutable bool kernel_available_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_GRAPH_FUSION_NORMALIZE_FUSION_H_
