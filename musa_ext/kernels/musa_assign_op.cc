// musa_assign_op.cc
#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "utils_op.h"
namespace tensorflow {
namespace musa {

// 在 .mu 中实现
template <typename T>
void LaunchAssignCopy(const T* src, T* dst, int64_t n, musaStream_t stream);

template <typename T>
class MusaAssignOp : public MusaOpKernel {
 public:
  explicit MusaAssignOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("validate_shape", &validate_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    // inputs:
    //   0: ref (Ref(T))
    //   1: value (T)
    const Tensor& value = ctx->input(1);

    // outputs:
    //   0: ref (Ref(T))  —— 直接把 ref 输入转发到 ref 输出
    ctx->forward_ref_input_to_ref_output(0, 0);

    // use_locking=True => 让 TF 内部帮你加锁，所以 lock_held=false
    // use_locking=False => 不加锁，lock_held=true（表示“锁已经持有”，TF
    // 不再加锁）
    const bool lock_held = !use_locking_;

    // mutable_input 用于获取/修改 ref tensor
    Tensor ref_tensor = ctx->mutable_input(0, lock_held);

    const bool ref_initialized = ref_tensor.IsInitialized();
    const bool same_shape =
        ref_initialized && ref_tensor.shape().IsSameSize(value.shape());

    // validate_shape=True：如果 ref 已初始化，则必须同 shape
    if (validate_shape_ && ref_initialized) {
      OP_REQUIRES(
          ctx, same_shape,
          errors::InvalidArgument(
              "Assign requires shapes to match when validate_shape=true. "
              "ref shape: ",
              ref_tensor.shape().DebugString(),
              ", value shape: ", value.shape().DebugString()));
    }

    // 如果 ref 未初始化，或者 validate_shape=false 且 shape 不同：允许 ref 变形
    // 对 ref 变量的正确做法是分配新 Tensor 并 replace_ref_input
    if (!ref_initialized || (!validate_shape_ && !same_shape)) {
      Tensor new_ref;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(value.dtype(), value.shape(), &new_ref));
      ctx->replace_ref_input(0, new_ref, lock_held);

      // replace 之后重新拿到 ref_tensor（指向新 buffer）
      ref_tensor = ctx->mutable_input(0, lock_held);
    }

    OP_REQUIRES(ctx, ref_tensor.NumElements() == value.NumElements(),
                errors::Internal(
                    "Assign: element count mismatch after shape handling."));

    const int64_t n = value.NumElements();
    if (n == 0) return;

    // 取 stream：如果你们不是 GetStream()，改成你们 addn 里的那一行即可
    musaStream_t stream = GetDeviceByCtx(ctx)->GetStream();

    const T* src = value.flat<T>().data();
    T* dst = ref_tensor.flat<T>().data();

    LaunchAssignCopy<T>(src, dst, n, stream);
  }

 private:
  bool use_locking_ = true;
  bool validate_shape_ = true;
};

// ---------------- Register ----------------
// Assign 的 TypeConstraint 是 "T"
#define REGISTER_MUSA_ASSIGN(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Assign").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaAssignOp<TYPE>)

// 你之前 AddN 测试/实现里用到的 4 种类型为主
REGISTER_MUSA_ASSIGN(float);
REGISTER_MUSA_ASSIGN(double);
REGISTER_MUSA_ASSIGN(Eigen::half);
REGISTER_MUSA_ASSIGN(bfloat16);

#undef REGISTER_MUSA_ASSIGN

}  // namespace musa
}  // namespace tensorflow