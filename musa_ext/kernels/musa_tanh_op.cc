#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaTahnOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Unary::Mode::TANH), "Set Tanh", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt), "Tanh Forward Run", ctx);
  }
};

REGISTER_KERNEL_BUILDER(Name("Tanh").Device("MUSA").TypeConstraint<float>("T"),
                        MusaTahnOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Tanh").Device("MUSA").TypeConstraint<Eigen::half>("T"),
    MusaTahnOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("Tanh").Device("MUSA").TypeConstraint<bfloat16>("T"),
    MusaTahnOp<bfloat16>);

REGISTER_KERNEL_BUILDER(Name("Tanh").Device("MUSA").TypeConstraint<double>("T"),
                        MusaTahnOp<double>);

}  // namespace musa
}  // namespace tensorflow
