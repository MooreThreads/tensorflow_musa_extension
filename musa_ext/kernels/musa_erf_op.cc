#include <cmath>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchErf(const T* src, T* dst, int n, musaStream_t stream);

template <typename T>
class MusaErfOp : public MusaOpKernel {
 public:
  explicit MusaErfOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    const T* input_ptr = input.flat<T>().data();
    T* output_ptr = output->flat<T>().data();
    const int64 num_elements = input.NumElements();

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    LaunchErf<T>(input_ptr, output_ptr, static_cast<int>(num_elements), stream);
  }
};

#define REGISTER_MUSA_ERF(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Erf").Device("MUSA").TypeConstraint<TYPE>("T"), MusaErfOp<TYPE>)

REGISTER_MUSA_ERF(float);
REGISTER_MUSA_ERF(double);
REGISTER_MUSA_ERF(Eigen::half);
REGISTER_MUSA_ERF(bfloat16);

#undef REGISTER_MUSA_ERF

}  // namespace musa
}  // namespace tensorflow
