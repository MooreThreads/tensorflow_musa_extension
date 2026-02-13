#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLogOp : public MusaOpKernel {
 public:
  explicit MusaLogOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    ::musa::dnn::Tensor mudnn_input = CreateMTensor(input, format_);
    ::musa::dnn::Tensor mudnn_output = CreateMTensor(*output, format_);

    ::musa::dnn::Unary log_op;
    log_op.SetMode(::musa::dnn::Unary::Mode::LOG);

    auto status = log_op.Run(handle, mudnn_output, mudnn_input);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Log execution failed. Status: ", (int)status));
  }
};

#define REGISTER_MUSA_LOG(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Log").Device("MUSA").TypeConstraint<TYPE>("T"), MusaLogOp<TYPE>)

REGISTER_MUSA_LOG(float);
REGISTER_MUSA_LOG(Eigen::half);
REGISTER_MUSA_LOG(bfloat16);

#undef REGISTER_MUSA_LOG

}  // namespace musa
}  // namespace tensorflow
