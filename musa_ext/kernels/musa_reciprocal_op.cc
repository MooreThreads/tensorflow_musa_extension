#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaReciprocalOp : public MusaOpKernel {
 public:
  explicit MusaReciprocalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());
    
    const Tensor& input = ctx->input(0);
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    
    if (input.NumElements() == 0) {
      return;
    }
    
    auto& handle = GetHandleByCtx(ctx);
    
    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);
    
    ::musa::dnn::Unary op;
    op.SetMode(::musa::dnn::Unary::Mode::RECIPROCAL);
    
    auto status = op.Run(handle, t_output, t_input);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Reciprocal execution failed."));
  }
};

#define REGISTER_MUSA_RECIPROCAL(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("Reciprocal").Device("MUSA").TypeConstraint<TYPE>("T"), MusaReciprocalOp<TYPE>);

REGISTER_MUSA_RECIPROCAL(float);
REGISTER_MUSA_RECIPROCAL(double);
REGISTER_MUSA_RECIPROCAL(Eigen::half);
REGISTER_MUSA_RECIPROCAL(bfloat16);
REGISTER_MUSA_RECIPROCAL(int32);
REGISTER_MUSA_RECIPROCAL(int64);

} // namespace musa
} // namespace tensorflow
