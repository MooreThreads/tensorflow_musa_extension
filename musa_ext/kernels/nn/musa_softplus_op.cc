#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchSoftplus(const T* input, T* output, int n, musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSoftplusOp : public MusaOpKernel {
 public:
  explicit MusaSoftplusOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    const int64_t n64 = y->NumElements();
    if (n64 == 0) return;

    OP_REQUIRES(ctx,
                n64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                errors::InvalidArgument(
                    "Softplus: tensor is too large, num_elements=", n64));

    const int n = static_cast<int>(n64);

    const T* x_ptr = x.flat<T>().data();
    T* y_ptr = y->flat<T>().data();

    auto* device = GetDeviceByCtx(ctx);
    auto stream = device->GetStream();

    LaunchSoftplus<T>(x_ptr, y_ptr, n, stream);
  }
};

#define REGISTER_MUSA_SOFTPLUS(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Softplus").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaSoftplusOp<TYPE>);

REGISTER_MUSA_SOFTPLUS(float);
REGISTER_MUSA_SOFTPLUS(Eigen::half);
REGISTER_MUSA_SOFTPLUS(bfloat16);

#undef REGISTER_MUSA_SOFTPLUS

}  // namespace musa
}  // namespace tensorflow