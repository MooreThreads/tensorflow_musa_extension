#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "utils_op.h"
#include "mu/device/musa_device.h"

// 声明 kernel 启动函数
namespace tensorflow {
namespace musa {
template <typename T>
void LaunchIsNan(const T* input, bool* output, int n, musaStream_t stream);
}  // namespace musa
}  // namespace tensorflow

namespace tensorflow {
namespace musa {

template <typename T>
class MusaIsNanOp : public MusaOpKernel {
 public:
  explicit MusaIsNanOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    const int64_t n64 = y->NumElements();
    if (n64 == 0) return;

    // TF Tensor 的 NumElements() 是 int64，这里 kernel 用 int
    OP_REQUIRES(ctx, n64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                errors::InvalidArgument("IsNan: tensor is too large, num_elements=", n64));

    const int n = static_cast<int>(n64);

    const T* x_ptr = x.flat<T>().data();
    bool* y_ptr = y->flat<bool>().data();

    auto* device = GetDeviceByCtx(ctx);
    auto stream = device->GetStream();

    LaunchIsNan<T>(x_ptr, y_ptr, n, stream);
  }
};

// 只注册 MUSA 设备上的 kernel；Op 本体（REGISTER_OP）由 TF Core 提供
#define REGISTER_MUSA_ISNAN(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(Name("IsNan")                              \
                              .Device(DEVICE_MTGPU)                 \
                              .TypeConstraint<TYPE>("T"),           \
                          MusaIsNanOp<TYPE>);

REGISTER_MUSA_ISNAN(float);
REGISTER_MUSA_ISNAN(double);
REGISTER_MUSA_ISNAN(Eigen::half);
REGISTER_MUSA_ISNAN(bfloat16);

#undef REGISTER_MUSA_ISNAN

}  // namespace musa
}  // namespace tensorflow
