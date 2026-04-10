#ifndef TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_LINEAR_LEAKYRELU_FUSION_H_
#define TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_LINEAR_LEAKYRELU_FUSION_H_

#include <string>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Computes: MatMul + BiasAdd/Add/AddV2 + LeakyRelu
class LinearLeakyReluFusion : public FusionPattern {
 public:
  LinearLeakyReluFusion() = default;
  ~LinearLeakyReluFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 100; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "LinearLeakyReluFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "LinearLeakyReluFusion kernel not available on this device";
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

#endif  // TENSORFLOW_MUSA_EXTENSION_MU_GRAPH_FUSION_LINEAR_LEAKYRELU_FUSION_H_
