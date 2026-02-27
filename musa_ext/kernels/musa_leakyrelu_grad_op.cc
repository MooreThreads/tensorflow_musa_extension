#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLeakyReluGradOp : public MusaOpKernel {
 public:
  explicit MusaLeakyReluGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& gradients = ctx->input(0);
    const Tensor& features = ctx->input(1);

    OP_REQUIRES(ctx, gradients.shape() == features.shape(),
                errors::InvalidArgument(
                    "Inputs to LeakyReluGrad must have the same shape. ",
                    "gradients: ", gradients.shape().DebugString(),
                    ", features: ", features.shape().DebugString()));

    Tensor* backprops = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, gradients.shape(), &backprops));

    if (gradients.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto mt_grads = CreateMTensor(gradients, format_);
    auto mt_feats = CreateMTensor(features, format_);
    auto mt_backprops = CreateMTensor(*backprops, format_);

    ::musa::dnn::Binary leaky_relu_grad_op;
    leaky_relu_grad_op.SetMode(::musa::dnn::Binary::Mode::LEAKY_RELU_BW);
    leaky_relu_grad_op.SetAlpha(alpha_);
    auto status =
        leaky_relu_grad_op.Run(handle, mt_backprops, mt_grads, mt_feats);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA LeakyReluGrad execution failed. Status: ",
                         (int)status));
  }

 private:
  float alpha_ = 0.2f;
};

#define REGISTER_MUSA_LEAKYRELU_GRAD(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LeakyReluGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLeakyReluGradOp<TYPE>)

REGISTER_MUSA_LEAKYRELU_GRAD(float);
REGISTER_MUSA_LEAKYRELU_GRAD(Eigen::half);
REGISTER_MUSA_LEAKYRELU_GRAD(bfloat16);

#undef REGISTER_MUSA_LEAKYRELU_GRAD

}  // namespace musa
}  // namespace tensorflow