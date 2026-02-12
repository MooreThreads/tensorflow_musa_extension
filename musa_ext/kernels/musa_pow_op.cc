#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "utils_op.h"

namespace tensorflow{
namespace musa{ 

template <typename T>
class MusaPowOp : public MusaOpKernel {
  // Compute the `x^y` for corresponding elements in x and y
  
  public:
    explicit MusaPowOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override { 
      const Tensor& x = ctx->input(0);
      const Tensor& y = ctx->input(1);

      const int dims0 = x.dims();
      const int dims1 = y.dims();
      // maybe add some dim check here?

      // The output should be the same as both inputs
      TensorShape output_shape = x.shape();

      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &out));

      // If either input is empty, return empty output
      if (in0.NumElements() == 0 || in1.NumElements() == 0 || output_shape.num_elements() == 0) {
        return;
      }

      auto& handle = GetHandleByCtx(ctx);// CreateMTensor 会自动处理 float/half/bfloat16 的类型映射
      mTensor tx = CreateMTensor(x, format_);
      mTensor ty = CreateMTensor(y, format_);
      mTensor t_out = CreateMTensor(*out, format_);
  
      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::POW);

      auto status = op.Run(handle, t_out, tx, ty);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Add execution failed."));
    }
}

#define REGISTER_KERNEL(TYPE)  \
  REGISTER_KERNLE_BUILDER(Name("Pow").Device("MUSA").TypeConstraint<TYPE>("T"), MusaPowOp<TYPE>);

REGISTER_MUSA_POW(float16);
REGISTER_MUSA_POW(float32);
REGISTER_MUSA_POW(float64);
REGISTER_MUSA_POW(int32);
REGISTER_MUSA_POW(int64);
REGISTER_MUSA_POW(complex64);
REGISTER_MUSA_POW(complex128);
    
}
}