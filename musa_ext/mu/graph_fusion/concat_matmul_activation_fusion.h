#ifndef TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_CONCAT_MATMUL_ACTIVATION_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_CONCAT_MATMUL_ACTIVATION_FUSION_H_

#include <string>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Computes:
//   ConcatV2 + MatMul + BiasAdd + Relu
//   ConcatV2 + MatMul + BiasAdd + LeakyRelu
class ConcatMatMulActivationFusion : public FusionPattern {
 public:
  ConcatMatMulActivationFusion() = default;
  ~ConcatMatMulActivationFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 125; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override {
    return "ConcatMatMulActivationFusion";
  }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "ConcatMatMulActivationFusion kernel not available on this device";
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

#endif  // TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_CONCAT_MATMUL_ACTIVATION_FUSION_H_
