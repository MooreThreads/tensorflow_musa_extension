#include "../utils_op.h"

namespace tensorflow {
namespace musa {
class MusaIdentityNOp : public MusaOpKernel {
 public:
  explicit MusaIdentityNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {}
};
}  // namespace musa
}  // namespace tensorflow