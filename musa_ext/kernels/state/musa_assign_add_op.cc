// musa_assign_add_op.cc
// MUSA implementation of AssignAdd operator
// Performs in-place addition: ref = ref + value

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace musa {

// ============================================================================
// External Kernel Launcher Declarations (implemented in
// musa_assign_add_kernel.mu) Using extern "C" linkage for MUSA kernels
// ============================================================================

extern "C" {
void LaunchAssignAddFloat(float* ref, const float* value, int64_t n,
                          musaStream_t stream);
void LaunchAssignAddDouble(double* ref, const double* value, int64_t n,
                           musaStream_t stream);
void LaunchAssignAddHalf(void* ref, const void* value, int64_t n,
                         musaStream_t stream);
void LaunchAssignAddBFloat16(void* ref, const void* value, int64_t n,
                             musaStream_t stream);
void LaunchAssignAddInt32(int* ref, const int* value, int64_t n,
                          musaStream_t stream);
void LaunchAssignAddInt64(int64_t* ref, const int64_t* value, int64_t n,
                          musaStream_t stream);
}

// ============================================================================
// AssignAdd Op Implementation
// ============================================================================

template <typename T>
class MusaAssignAddOp : public MusaOpKernel {
 public:
  explicit MusaAssignAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
  }

  // AssignAdd is a memory-bound operation with simple arithmetic
  // Mark as not expensive to enable inline scheduling
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override;

 private:
  bool use_locking_ = false;
};

// ============================================================================
// Template Specializations for Compute
// ============================================================================

template <>
void MusaAssignAddOp<float>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  LaunchAssignAddFloat(ref_tensor.flat<float>().data(),
                       value.flat<float>().data(), n, stream);
}

template <>
void MusaAssignAddOp<double>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  LaunchAssignAddDouble(ref_tensor.flat<double>().data(),
                        value.flat<double>().data(), n, stream);
}

template <>
void MusaAssignAddOp<Eigen::half>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  // Cast to void* for half type (matches MUSA half* ABI)
  LaunchAssignAddHalf(
      reinterpret_cast<void*>(ref_tensor.flat<Eigen::half>().data()),
      reinterpret_cast<const void*>(value.flat<Eigen::half>().data()), n,
      stream);
}

template <>
void MusaAssignAddOp<bfloat16>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  // Cast to void* for bfloat16 type (matches MUSA __mt_bfloat16* ABI)
  LaunchAssignAddBFloat16(
      reinterpret_cast<void*>(ref_tensor.flat<bfloat16>().data()),
      reinterpret_cast<const void*>(value.flat<bfloat16>().data()), n, stream);
}

template <>
void MusaAssignAddOp<int32>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  LaunchAssignAddInt32(ref_tensor.flat<int32>().data(),
                       value.flat<int32>().data(), n, stream);
}

template <>
void MusaAssignAddOp<int64>::Compute(OpKernelContext* ctx) {
  const Tensor& value = ctx->input(1);
  ctx->forward_ref_input_to_ref_output(0, 0);
  const bool lock_held = !use_locking_;
  Tensor ref_tensor = ctx->mutable_input(0, lock_held);

  OP_REQUIRES(ctx, ref_tensor.shape().IsSameSize(value.shape()),
              errors::InvalidArgument(
                  "AssignAdd: ref and value must have the same shape. "
                  "ref shape: ",
                  ref_tensor.shape().DebugString(),
                  ", value shape: ", value.shape().DebugString()));

  const int64_t n = value.NumElements();
  if (n == 0) return;

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  // Use reinterpret_cast for int64 type (tensorflow::int64 is long long,
  // int64_t is long)
  LaunchAssignAddInt64(
      reinterpret_cast<int64_t*>(ref_tensor.flat<int64>().data()),
      reinterpret_cast<const int64_t*>(value.flat<int64>().data()), n, stream);
}

// ============================================================================
// Kernel Registration
// ============================================================================

#define REGISTER_MUSA_ASSIGN_ADD(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AssignAdd").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaAssignAddOp<TYPE>)

REGISTER_MUSA_ASSIGN_ADD(float);
REGISTER_MUSA_ASSIGN_ADD(double);
REGISTER_MUSA_ASSIGN_ADD(Eigen::half);
REGISTER_MUSA_ASSIGN_ADD(bfloat16);
REGISTER_MUSA_ASSIGN_ADD(int32);
REGISTER_MUSA_ASSIGN_ADD(int64);

#undef REGISTER_MUSA_ASSIGN_ADD

}  // namespace musa
}  // namespace tensorflow
