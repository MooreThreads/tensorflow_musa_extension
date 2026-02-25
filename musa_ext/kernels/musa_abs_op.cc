#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaAbsOp : public MusaOpKernel {
 public:
  explicit MusaAbsOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mUnary unary_op;
    unary_op.SetMode(::musa::dnn::Unary::Mode::ABS);

    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    auto status = unary_op.Run(handle, mt_output, mt_input);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Abs execution failed. Status: ", (int)status));
  }
};

#define REGISTER_MUSA_ABS(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Abs").Device("MUSA").TypeConstraint<TYPE>("T"), MusaAbsOp<TYPE>)

REGISTER_MUSA_ABS(float);
REGISTER_MUSA_ABS(Eigen::half);
REGISTER_MUSA_ABS(bfloat16);
REGISTER_MUSA_ABS(double);
REGISTER_MUSA_ABS(int32);
REGISTER_MUSA_ABS(int64);

#undef REGISTER_MUSA_ABS

}  // namespace musa
}  // namespace tensorflow
