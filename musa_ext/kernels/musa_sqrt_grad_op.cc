#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSqrtGradOp : public MusaOpKernel {
 public:
  explicit MusaSqrtGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& y = ctx->input(0);
    const Tensor& dy = ctx->input(1);

    BCast bcast(BCast::FromShape(y.shape()), BCast::FromShape(dy.shape()),
                false);

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for SqrtGrad: ",
                                        "y: ", y.shape().DebugString(),
                                        ", dy: ", dy.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.result_shape());
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &dx));

    if (output_shape.num_elements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_y = CreateMTensor(y, format_);
    mTensor t_dy = CreateMTensor(dy, format_);
    mTensor t_dx = CreateMTensor(*dx, format_);

    Tensor temp_div;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(y.dtype(), output_shape, &temp_div));
    mTensor t_temp_div = CreateMTensor(temp_div, format_);

    ::musa::dnn::Binary binary_op;

    binary_op.SetMode(::musa::dnn::Binary::Mode::TRUEDIV);

    auto status = binary_op.Run(handle, t_temp_div, t_dy, t_y);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA TRUEDIV failed. muDNN might not support "
                                 "Double (FP64) for this op."));

    Tensor half_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(y.dtype(), TensorShape({1}), &half_tensor));
    T half_host = static_cast<T>(0.5);

    mStatus copy_status = MusaMemcpyH2D(
        const_cast<T*>(half_tensor.flat<T>().data()), &half_host, sizeof(T));
    OP_REQUIRES(ctx, copy_status == mStatus::SUCCESS,
                errors::Internal("MUSA H2D copy failed for constant 0.5"));

    mTensor t_half = CreateMTensor(half_tensor, format_);

    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    status = binary_op.Run(handle, t_dx, t_temp_div, t_half);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA MUL failed in SqrtGrad."));
  }
};

#define REGISTER_MUSA_SQRT_GRAD(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SqrtGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSqrtGradOp<TYPE>)

REGISTER_MUSA_SQRT_GRAD(float);
REGISTER_MUSA_SQRT_GRAD(Eigen::half);
REGISTER_MUSA_SQRT_GRAD(bfloat16);

#undef REGISTER_MUSA_SQRT_GRAD

}  // namespace musa
}  // namespace tensorflow
