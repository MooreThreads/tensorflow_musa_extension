#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaReluOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    mUnary op;
    op.SetMode(::musa::dnn::Unary::Mode::RELU);

    auto status = op.Run(handle, t_output, t_input);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA muDNN Unary ReLU execution failed. Status: ",
                         (int)status));
  }
};

REGISTER_KERNEL_BUILDER(Name("Relu").Device("MUSA").TypeConstraint<float>("T"),
                        MusaReluOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device("MUSA").TypeConstraint<Eigen::half>("T"),
    MusaReluOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device("MUSA").TypeConstraint<bfloat16>("T"),
    MusaReluOp<bfloat16>);

REGISTER_KERNEL_BUILDER(Name("Relu").Device("MUSA").TypeConstraint<double>("T"),
                        MusaReluOp<double>);

}  // namespace musa
}  // namespace tensorflow
