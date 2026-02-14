#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLog1pOp : public MusaOpKernel {
 public:
  explicit MusaLog1pOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    op.SetMode(::musa::dnn::Unary::Mode::LOG1P);

    auto status = op.Run(handle, t_output, t_input);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Log1p execution failed."));
  }
};

#define REGISTER_MUSA_LOG1P(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Log1p").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLog1pOp<TYPE>);

REGISTER_MUSA_LOG1P(float);
REGISTER_MUSA_LOG1P(double);
REGISTER_MUSA_LOG1P(Eigen::half);
REGISTER_MUSA_LOG1P(bfloat16);
REGISTER_MUSA_LOG1P(int32);
REGISTER_MUSA_LOG1P(int64);

}  // namespace musa
}  // namespace tensorflow
