// musa_stop_gradient_op.cc
#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

#include <limits>

namespace tensorflow {
namespace musa {

// forward: y = x
// 这里用 memcpy（device-to-device）实现；也允许 output 复用 input buffer 时直接跳过。
template <typename T>
void LaunchStopGradient(const T* input, T* output, int n, musaStream_t stream) {
  if (n <= 0) return;
  if (input == output) return;

  // 假设 MUSA runtime 提供 musaMemcpyAsync / musaMemcpyDeviceToDevice
  // 如果你们环境的 API 名称不同（比如 muMemcpyAsync），改这里一行即可。
  musaError_t err =
      musaMemcpyAsync(output, input, sizeof(T) * n, musaMemcpyDeviceToDevice,
                      stream);
  // 这里不使用未知的项目宏（如 MUSA_CHECK），而是用最朴素的 TF 错误处理方式：
  // 注意：LaunchStopGradient 本身没有 ctx，所以这里用 CHECK/abort 风格不优雅；
  // 更推荐在 Compute 里做 OP_REQUIRES 检查（下方已做）。
  (void)err;
}

}  // namespace musa
}  // namespace tensorflow

namespace tensorflow {
namespace musa {

template <typename T>
class MusaStopGradientOp : public MusaOpKernel {
 public:
  explicit MusaStopGradientOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);

    // StopGradient forward 等价于 Identity：尽量复用输入 buffer（减少一次分配/拷贝）
    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(),
                                                              &y));

    const int64_t n64 = y->NumElements();
    if (n64 == 0) return;

    OP_REQUIRES(ctx,
                n64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                errors::InvalidArgument(
                    "StopGradient: tensor is too large, num_elements=", n64));

    const int n = static_cast<int>(n64);

    const T* x_ptr = x.flat<T>().data();
    T* y_ptr = y->flat<T>().data();

    // 如果 forward_input_or_allocate_output 复用了 input，这里指针可能相同，直接 return
    if (x_ptr == y_ptr) return;

    auto* device = GetDeviceByCtx(ctx);
    auto stream = device->GetStream();

    // 调用 D2D memcpy（异步 enqueue 到当前 stream）
    musaError_t err =
        musaMemcpyAsync(y_ptr, x_ptr, sizeof(T) * n, musaMemcpyDeviceToDevice,
                        stream);
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("StopGradient musaMemcpyAsync failed, err=",
                                 static_cast<int>(err)));
  }
};

// TF 官方 op 名称是 "StopGradient"（forward = Identity，backward 由梯度注册实现截断）
#define REGISTER_MUSA_STOP_GRADIENT(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("StopGradient").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaStopGradientOp<TYPE>);

// 按 softplus 风格：先注册常用浮点类型（你需要更多类型就继续加）
REGISTER_MUSA_STOP_GRADIENT(float);
REGISTER_MUSA_STOP_GRADIENT(Eigen::half);
REGISTER_MUSA_STOP_GRADIENT(bfloat16);

#undef REGISTER_MUSA_STOP_GRADIENT

}  // namespace musa
}  // namespace tensorflow