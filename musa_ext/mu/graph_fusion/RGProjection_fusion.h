#pragma once

#include <string>
#include <vector>

#include "mu/graph_fusion/fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Computes: MatMul(Relu(BiasAdd(x, b)), w)
// Actual dataflow order: BiasAdd -> Relu -> MatMul

class BiasAddReluMatMulFusion : public FusionPattern {
 public:
  BiasAddReluMatMulFusion() = default;
  ~BiasAddReluMatMulFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 97; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "BiasAddReluMatMulFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "BiasAddReluMatMulFusion kernel not available on this device";
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
