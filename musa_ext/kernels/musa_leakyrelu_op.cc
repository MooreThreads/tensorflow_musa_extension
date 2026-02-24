#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLeakyReluOp : public MusaOpKernel {
 public:
  explicit MusaLeakyReluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU),
                  "Set LeakyRelu", ctx);
    MTOP_CHECK_OK(op.SetAlpha(static_cast<double>(alpha_)),
                  "Set LeakyRelu alpha", ctx);

    MTOP_CHECK_OK_RUN(op.Run(handle, t_output, t_input), "LeakyRelu Run", ctx);
  }

 private:
  float alpha_ = 0.2f;
};

// Register the LeakyRelu operator
#define REGISTER_MUSA_LEAKYRELU(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("LeakyRelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLeakyReluOp<TYPE>)

REGISTER_MUSA_LEAKYRELU(float);
REGISTER_MUSA_LEAKYRELU(Eigen::half);
REGISTER_MUSA_LEAKYRELU(bfloat16);
REGISTER_MUSA_LEAKYRELU(double);

#undef REGISTER_MUSA_LEAKYRELU

}  // namespace musa
}  // namespace tensorflow