#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void MusaNegKernelLauncher(const void* in, void* out, int size,
                           musaStream_t stream);

template <typename T>
class MusaNegOp : public MusaOpKernel {
 public:
  explicit MusaNegOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    MusaNegKernelLauncher<T>(input.tensor_data().data(),
                             const_cast<char*>(output->tensor_data().data()),
                             input.NumElements(), stream);
  }
};

#define REGISTER_MUSA_NEG(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Neg").Device("MUSA").TypeConstraint<TYPE>("T"), MusaNegOp<TYPE>)

REGISTER_MUSA_NEG(float);
REGISTER_MUSA_NEG(double);
REGISTER_MUSA_NEG(int32);
REGISTER_MUSA_NEG(int64);
REGISTER_MUSA_NEG(Eigen::half);
REGISTER_MUSA_NEG(bfloat16);

#undef REGISTER_MUSA_NEG

}  // namespace musa
}  // namespace tensorflow
