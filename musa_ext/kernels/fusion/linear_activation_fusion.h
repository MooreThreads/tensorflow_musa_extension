#include <string>
#include <vector>

#include "fusion_pattern_manager.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

// Computes: MatMul + BiasAdd + Activation

class LinearActivationFusion : public FusionPattern {
 public:
  LinearActivationFusion() = default;
  ~LinearActivationFusion() override = default;

  FusionMatchResult Match(const GraphDef& graph,
                          int start_node_idx) const override;

  Status Apply(GraphDef* graph,
               const FusionMatchResult& match_result) const override;

  int GetPriority() const override { return 120; }

  bool IsKernelAvailable() const override;

  std::string GetName() const override { return "LinearActivationFusion"; }

  std::string GetFallbackReason() const override {
    if (!kernel_available_) {
      return "LinearActivationFusion kernel not available on this device";
    }
    return "";
  }

 private:
  // Kernel availability flag
  mutable bool kernel_available_ = true;
  mutable bool kernel_checked_ = false;
};

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
