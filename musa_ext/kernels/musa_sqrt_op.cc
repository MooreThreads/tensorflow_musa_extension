#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSqrtOp : public MusaOpKernel {
 public:
  explicit MusaSqrtOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Sqrt is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    ::musa::dnn::Tensor mudnn_input = CreateMTensor(input);
    ::musa::dnn::Tensor mudnn_output = CreateMTensor(*output);

    Tensor zero_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), input.shape(), &zero_tensor));
    Tensor clamped_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input.dtype(), input.shape(), &clamped_tensor));

    ::musa::dnn::Tensor mudnn_zero = CreateMTensor(zero_tensor);
    ::musa::dnn::Tensor mudnn_clamped = CreateMTensor(clamped_tensor);

    ::musa::dnn::Fill fill_op;
    fill_op.SetValue(0.0f);
    auto status = fill_op.Run(handle, mudnn_zero);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Fill execution failed. Status: ", (int)status));

    ::musa::dnn::Binary max_op;
    max_op.SetMode(::musa::dnn::Binary::Mode::MAX);
    status = max_op.Run(handle, mudnn_clamped, mudnn_input, mudnn_zero);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Maximum execution failed. Status: ",
                                 (int)status));

    ::musa::dnn::Unary sqrt_op;
    sqrt_op.SetMode(::musa::dnn::Unary::Mode::SQRT);

    status = sqrt_op.Run(handle, mudnn_output, mudnn_clamped);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Sqrt execution failed. Status: ", (int)status));
  }
};

REGISTER_KERNEL_BUILDER(Name("Sqrt").Device("MUSA").TypeConstraint<float>("T"),
                        MusaSqrtOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Sqrt").Device("MUSA").TypeConstraint<Eigen::half>("T"),
    MusaSqrtOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("Sqrt").Device("MUSA").TypeConstraint<Eigen::bfloat16>("T"),
    MusaSqrtOp<Eigen::bfloat16>);

}  // namespace musa
}  // namespace tensorflow
