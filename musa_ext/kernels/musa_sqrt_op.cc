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

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    
    if (input.NumElements() == 0) return;
    
    auto& handle = GetHandleByCtx(ctx);
    
    ::musa::dnn::Tensor mudnn_input = CreateMTensor(input);
    ::musa::dnn::Tensor mudnn_output = CreateMTensor(*output);
    
    ::musa::dnn::Unary sqrt_op;
    sqrt_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    
    auto status = sqrt_op.Run(handle, mudnn_output, mudnn_input);
    
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
               errors::Internal("MUSA Sqrt execution failed. Status: ", (int)status));
  }
};

REGISTER_KERNEL_BUILDER(Name("Sqrt")
                            .Device("MUSA")
                            .TypeConstraint<float>("T"),
                        MusaSqrtOp<float>);

REGISTER_KERNEL_BUILDER(Name("Sqrt")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::half>("T"),
                        MusaSqrtOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(Name("Sqrt")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::bfloat16>("T"),
                        MusaSqrtOp<Eigen::bfloat16>);
} 
}
