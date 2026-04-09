#include <cstdlib>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  // Multiply is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for Mul: ", in0.shape().DebugString(),
                    " and ", in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;
    if (in0.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {0};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else if (in1.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {1};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    }

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL);

    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Multiply execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_MULTIPLY(TYPE)                        \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("Mul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMultiplyOp<TYPE>);

REGISTER_MUSA_MULTIPLY(float);
REGISTER_MUSA_MULTIPLY(Eigen::half);
REGISTER_MUSA_MULTIPLY(bfloat16);
REGISTER_MUSA_MULTIPLY(int32);
REGISTER_MUSA_MULTIPLY(int64);

#undef REGISTER_MUSA_MULTIPLY

}  // namespace musa
}  // namespace tensorflow
