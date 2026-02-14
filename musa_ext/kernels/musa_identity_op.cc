#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/version.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaIdentityOp : public OpKernel {
 public:
  explicit MusaIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    int num_outputs = context->num_outputs();
    int num_inputs = context->num_inputs();

    if (num_outputs > 0) {
      if (num_inputs > 0) {
        for (int i = 0; i < num_outputs && i < num_inputs; ++i) {
          context->set_output(i, context->input(i));
        }
      } else {
        VLOG(1) << ">>>>> [MUSA] Identity/Arg with 0 inputs, skipping manual "
                   "set_output";
      }
    }
  }
};

#define REGISTER_MUSA_BASE_OPS(type)                              \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Identity").Device("MUSA").TypeConstraint<type>("T"),  \
      MusaIdentityOp);                                            \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("IdentityN").Device("MUSA").TypeConstraint<type>("T"), \
      MusaIdentityOp);                                            \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Snapshot").Device("MUSA").TypeConstraint<type>("T"),  \
      MusaIdentityOp);
REGISTER_MUSA_BASE_OPS(float);
REGISTER_MUSA_BASE_OPS(double);
REGISTER_MUSA_BASE_OPS(Eigen::half);
REGISTER_MUSA_BASE_OPS(int32);
REGISTER_MUSA_BASE_OPS(int64);
REGISTER_MUSA_BASE_OPS(bool);

}  // namespace musa
}  // namespace tensorflow
