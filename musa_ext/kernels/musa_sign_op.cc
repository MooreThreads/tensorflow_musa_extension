#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void MusaSignKernelLauncher(const T* input, T* output, int64_t size);

template <typename T>
class MusaSignOp : public MusaOpKernel {
 public:
  explicit MusaSignOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    
    const int64_t size = input.NumElements();
    
    if (size == 0) return;
    
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();
    
    MusaSignKernelLauncher<T>(input.flat<T>().data(),
                              output->flat<T>().data(),
                              size);
    
    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA Sign kernel failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

#define REGISTER_MUSA_SIGN(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("Sign")                              \
                              .Device("MUSA")                       \
                              .TypeConstraint<TYPE>("T"),           \
                          MusaSignOp<TYPE>)

REGISTER_MUSA_SIGN(float);
REGISTER_MUSA_SIGN(int32);
REGISTER_MUSA_SIGN(long long);
REGISTER_MUSA_SIGN(bfloat16);
REGISTER_MUSA_SIGN(Eigen::half);

#undef REGISTER_MUSA_SIGN

}  // namespace musa
}  // namespace tensorflow