#include "../array/musa_fill_functor.h"
#include "../utils_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

/**
 * MusaSigmoidCalibrationOp
 *
 * Performs fusion: S / (S + Scale * (1 - S))
 * where S = Sigmoid(x)
 *
 * This implements the specific activation logic from the given graph.
 */
template <typename T>
class MusaSigmoidCalibrationOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidCalibrationOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& scale = ctx->input(1);

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    mTensor in_mt = CreateMTensor(input);
    mTensor scale_mt = CreateMTensor(scale);

    // 1. Sigmoid
    Tensor sigmoid_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), input.shape(), &sigmoid_output));
    mTensor sigmoid_mt = CreateMTensor(sigmoid_output);
    mUnary op;
    MTOP_CHECK_OK(op.SetMode(mUnary::Mode::SIGMOID), "Set Sigmoid", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, sigmoid_mt, in_mt), "Sigmoid Forward Run",
                      ctx);

    // 2. Compute 1 - S
    Tensor sub_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), input.shape(), &sub_output));
    mTensor sub_mt = CreateMTensor(sub_output);
    Tensor ones;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(input.dtype(), input.shape(), &ones));
    mTensor ones_mt = CreateMTensor(ones);
    SetOneFunctor::Compute<T>(ctx, &ones_mt);
    mBinary sub_op;
    MTOP_CHECK_OK(sub_op.SetMode(mBinary::Mode::SUB), "Set Sub", ctx);
    MTOP_CHECK_OK_RUN(sub_op.Run(handle, sub_mt, ones_mt, sigmoid_mt),
                      "Sub Run", ctx);

    // 3. Compute Scale * (1 - S)
    Tensor scale_mul_output;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                                           &scale_mul_output));
    mTensor scale_mul_mt = CreateMTensor(scale_mul_output);
    mBinary mul_op;
    MTOP_CHECK_OK(mul_op.SetMode(mBinary::Mode::MUL), "Set Mul", ctx);
    MTOP_CHECK_OK_RUN(mul_op.Run(handle, scale_mul_mt, scale_mt, sub_mt),
                      "Mul Run", ctx);

    // 4. Compute S + Scale * (1 - S)
    Tensor denom_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), input.shape(), &denom_output));
    mTensor denom_mt = CreateMTensor(denom_output);
    mBinary add_op;
    MTOP_CHECK_OK(add_op.SetMode(mBinary::Mode::ADD), "Set Add", ctx);
    MTOP_CHECK_OK_RUN(add_op.Run(handle, denom_mt, sigmoid_mt, scale_mul_mt),
                      "Add Run", ctx);

    // 5. Compute S / (S + Scale * (1 - S))
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    mTensor out_mt = CreateMTensor(*output);
    mBinary div_op;
    MTOP_CHECK_OK(div_op.SetMode(mBinary::Mode::DIV), "Set Div", ctx);
    MTOP_CHECK_OK_RUN(div_op.Run(handle, out_mt, sigmoid_mt, denom_mt),
                      "Div Run", ctx);
  }
};

#define REGISTER_MUSA_SIGMOID_CALIBRATION(type)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MusaSigmoidCalibration").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSigmoidCalibrationOp<type>);

REGISTER_MUSA_SIGMOID_CALIBRATION(float);
REGISTER_MUSA_SIGMOID_CALIBRATION(Eigen::half);
REGISTER_MUSA_SIGMOID_CALIBRATION(bfloat16);

}  // namespace musa
}  // namespace tensorflow
