#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSigmoidOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Unary::Mode::SIGMOID), "Set Sigmoid",
                  ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt), "Sigmoid Forward Run",
                      ctx);
  }
};

template <typename T>
class MusaSigmoidGradOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& y = ctx->input(0);
    const Tensor& dy = ctx->input(1);

    Tensor* dz = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &dz));

    if (y.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto y_mt = CreateMTensor(y, format_);
    auto dy_mt = CreateMTensor(dy, format_);
    auto dz_mt = CreateMTensor(*dz, format_);

    ::musa::dnn::Binary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::SIGMOID_BW),
                  "Set Sigmoid_BW", ctx);

    MTOP_CHECK_OK_RUN(op.Run(handle, dz_mt, dy_mt, y_mt),
                      "Sigmoid_BW Native Run", ctx);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Sigmoid").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("SigmoidGrad").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidGradOp<float>);

}  // namespace musa
}  // namespace tensorflow
