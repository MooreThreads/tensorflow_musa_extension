#ifndef TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_TWO_LAYER_CONCAT_MATMUL_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_TWO_LAYER_CONCAT_MATMUL_FUSION_H_

#include <string>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Computes:
//   ConcatV2 + MatMul + BiasAdd + Relu + MatMul + BiasAdd
//   ConcatV2 + MatMul + BiasAdd + LeakyRelu + MatMul + BiasAdd
class TwoLayerConcatMatMulFusion : public FusionPattern {
 public:
  TwoLayerConcatMatMulFusion() = default;
  ~TwoLayerConcatMatMulFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 140; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override {
    return "TwoLayerConcatMatMulFusion";
  }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "TwoLayerConcatMatMulFusion kernel not available on this device";
    }
    return "";
  }

 private:
  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_TWO_LAYER_CONCAT_MATMUL_FUSION_H_
