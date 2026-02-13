#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaReluGradOp : public MusaOpKernel {
 public:
  explicit MusaReluGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& gradients = ctx->input(0);
    const Tensor& features = ctx->input(1);

    OP_REQUIRES(ctx, gradients.shape() == features.shape(),
                errors::InvalidArgument(
                    "Inputs to ReluGrad must have the same shape. ",
                    "gradients: ", gradients.shape().DebugString(),
                    ", features: ", features.shape().DebugString()));

    Tensor* backprops = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, gradients.shape(), &backprops));

    if (gradients.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto mt_grads = CreateMTensor(gradients, format_);
    auto mt_feats = CreateMTensor(features, format_);
    auto mt_backprops = CreateMTensor(*backprops, format_);

    ::musa::dnn::Binary relu_grad_op;
    relu_grad_op.SetMode(::musa::dnn::Binary::Mode::LEAKY_RELU_BW);
    auto status = relu_grad_op.Run(handle, mt_backprops, mt_grads, mt_feats);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA ReluGrad execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_RELU_GRAD(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ReluGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaReluGradOp<TYPE>)

REGISTER_MUSA_RELU_GRAD(float);
REGISTER_MUSA_RELU_GRAD(Eigen::half);
REGISTER_MUSA_RELU_GRAD(bfloat16);

#undef REGISTER_MUSA_RELU_GRAD

}  // namespace musa
}  // namespace tensorflow
