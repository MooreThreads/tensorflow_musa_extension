#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaStopGradientOp : public MusaOpKernel {
 public:
  explicit MusaStopGradientOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, input);
    }
  }

  bool IsExpensive() override { return false; }
};

#define REGISTER_MUSA_STOP_GRADIENT(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("StopGradient").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaStopGradientOp<TYPE>)

REGISTER_MUSA_STOP_GRADIENT(float);
REGISTER_MUSA_STOP_GRADIENT(double);
REGISTER_MUSA_STOP_GRADIENT(Eigen::half);
REGISTER_MUSA_STOP_GRADIENT(int32);
REGISTER_MUSA_STOP_GRADIENT(int64);
REGISTER_MUSA_STOP_GRADIENT(bfloat16);

#undef REGISTER_MUSA_STOP_GRADIENT

}  // namespace musa
}  // namespace tensorflow
