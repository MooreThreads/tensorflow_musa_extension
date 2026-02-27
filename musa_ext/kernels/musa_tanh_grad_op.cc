#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaTanhGradOp : public MusaOpKernel {
 public:
  explicit MusaTanhGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

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
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::TANH_BW), "Set Tanh_BW",
                  ctx);

    MTOP_CHECK_OK_RUN(op.Run(handle, dz_mt, dy_mt, y_mt), "Tanh_BW Native Run",
                      ctx);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TanhGrad").Device("MUSA").TypeConstraint<float>("T"),
    MusaTanhGradOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("TanhGrad").Device("MUSA").TypeConstraint<Eigen::half>("T"),
    MusaTanhGradOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("TanhGrad").Device("MUSA").TypeConstraint<bfloat16>("T"),
    MusaTanhGradOp<bfloat16>);

REGISTER_KERNEL_BUILDER(
    Name("TanhGrad").Device("MUSA").TypeConstraint<double>("T"),
    MusaTanhGradOp<double>);

}  // namespace musa
}  // namespace tensorflow