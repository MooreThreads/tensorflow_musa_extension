#ifndef TENSORFLOW_MUSA_MU_OPTIMIZER_MUSA_REMAPPER_OPTIMIZER_H_
#define TENSORFLOW_MUSA_MU_OPTIMIZER_MUSA_REMAPPER_OPTIMIZER_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class MusaOptimizationPass : public GraphOptimizationPass {
 public:
  MusaOptimizationPass() = default;
  ~MusaOptimizationPass() override = default;

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  Node* FindNode(Graph* graph, const std::string& name);
  int CountConsumers(Node* node);
};

void ForceMusaOptimizationPassRegistration();

}  // namespace tensorflow

#endif
