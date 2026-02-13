#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaRsqrtOp : public MusaOpKernel {
 public:
  explicit MusaRsqrtOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      auto in_mt = CreateMTensor(input, format_);
      auto out_mt = CreateMTensor(*output, format_);
      auto& h = GetHandleByCtx(context);

      ::musa::dnn::Unary op;
      op.SetMode(::musa::dnn::Unary::Mode::RSQRT);

      auto status = op.Run(h, out_mt, in_mt);
      OP_REQUIRES(context, status == mStatus::SUCCESS,
                  errors::Internal("muDNN Rsqrt Forward Run failed"));
    }
  }
};

template <typename T>
class MusaRsqrtGradOp : public MusaOpKernel {
 public:
  explicit MusaRsqrtGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& y = context->input(0);
    const Tensor& dy = context->input(1);

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, y.shape(), &dx));

    if (y.NumElements() > 0) {
      auto y_mt = CreateMTensor(y, format_);
      auto dy_mt = CreateMTensor(dy, format_);
      auto dx_mt = CreateMTensor(*dx, format_);
      auto& h = GetHandleByCtx(context);

      ::musa::dnn::Binary op;
      auto status = op.SetMode(::musa::dnn::Binary::Mode::RSQRT_BW);
      OP_REQUIRES(context, status == mStatus::SUCCESS,
                  errors::Internal("muDNN Binary SetMode(RSQRT_BW) failed"));

      status = op.Run(h, dx_mt, y_mt, dy_mt);

      OP_REQUIRES(context, status == mStatus::SUCCESS,
                  errors::Internal("muDNN Rsqrt Backward Run failed. "
                                   "Check if the input shapes y: ",
                                   y.shape().DebugString(),
                                   " and dy: ", dy.shape().DebugString(),
                                   " are compatible."));
    }
  }
};

#define REGISTER_MUSA_RSQRT_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Rsqrt").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),     \
      MusaRsqrtOp<type>);                                               \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("RsqrtGrad").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaRsqrtGradOp<type>);

REGISTER_MUSA_RSQRT_KERNELS(float);
REGISTER_MUSA_RSQRT_KERNELS(double);
REGISTER_MUSA_RSQRT_KERNELS(Eigen::half);
REGISTER_MUSA_RSQRT_KERNELS(bfloat16);
REGISTER_MUSA_RSQRT_KERNELS(int32);
REGISTER_MUSA_RSQRT_KERNELS(int64);

#undef REGISTER_MUSA_RSQRT_KERNELS

}  // namespace musa
}  // namespace tensorflow