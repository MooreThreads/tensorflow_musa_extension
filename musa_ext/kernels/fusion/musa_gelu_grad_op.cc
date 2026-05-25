#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGeluGradOp : public MusaOpKernel {
 public:
  explicit MusaGeluGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dy = ctx->input(0);
    const Tensor& x = ctx->input(1);

    OP_REQUIRES(ctx, dy.shape() == x.shape(),
                errors::InvalidArgument(
                    "dy and x must have the same shape. dy: ",
                    dy.shape().DebugString(), ", x: ", x.shape().DebugString()));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));

    if (x.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto mt_dy = CreateMTensor(dy, format_);
    auto mt_x = CreateMTensor(x, format_);
    auto mt_dx = CreateMTensor(*dx, format_);

    ::musa::dnn::Binary op;
    const BINARY_MODE mode = approximate_ ? BINARY_MODE::GELU_TANH_BW
                                          : BINARY_MODE::GELU_NONE_BW;
    MTOP_CHECK_OK(op.SetMode(mode), "Set GELU_BW Mode", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, mt_dx, mt_dy, mt_x), "GELU_BW Run", ctx);
  }

 private:
  bool approximate_;
};

#define REGISTER_MUSA_GELU_GRAD(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MusaGeluGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaGeluGradOp<TYPE>);

REGISTER_MUSA_GELU_GRAD(float);
REGISTER_MUSA_GELU_GRAD(double);
REGISTER_MUSA_GELU_GRAD(Eigen::half);
REGISTER_MUSA_GELU_GRAD(bfloat16);

#undef REGISTER_MUSA_GELU_GRAD

}  // namespace musa

REGISTER_OP("MusaGeluGrad")
    .Input("dy: T")
    .Input("x: T")
    .Output("dx: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("approximate: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

}  // namespace tensorflow
