// musa_assign_variable_op.cc
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "utils_op.h"
#include "mu/device/musa_device.h"

namespace tensorflow {
namespace musa {

// 在 .cu 中实现
template <typename T>
void LaunchAssignCopy(const T* src, T* dst, int64_t n, musaStream_t stream);

// TF2: AssignVariableOp(resource, value)  无输出
template <typename T>
class MusaAssignVariableOp : public MusaOpKernel {
 public:
  explicit MusaAssignVariableOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("validate_shape", &validate_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(1);

    // resource handle 通常在 Host
    const ResourceHandle& handle = HandleFromInput(ctx, 0);

    Var* var = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &var));
    core::ScopedUnref unref(var);

    // TF2 resource variable: 必须加锁
    mutex_lock ml(*var->mu());

    Tensor* var_tensor = var->tensor();
    const bool var_initialized = var_tensor->IsInitialized();
    const bool same_shape =
        var_initialized && var_tensor->shape().IsSameSize(value.shape());

    // validate_shape=true: 已初始化时必须 shape 一致
    if (validate_shape_ && var_initialized) {
      OP_REQUIRES(ctx, same_shape,
                  errors::InvalidArgument(
                      "AssignVariableOp requires shapes to match when validate_shape=true. "
                      "var shape: ",
                      var_tensor->shape().DebugString(), ", value shape: ",
                      value.shape().DebugString()));
    }

    // 未初始化，或 validate_shape=false 且 shape 不同：允许 reshape -> 重新分配
    if (!var_initialized || (!validate_shape_ && !same_shape)) {
      Tensor new_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(value.dtype(), value.shape(), &new_tensor));
      *var_tensor = new_tensor;
    }

    OP_REQUIRES(ctx, var_tensor->NumElements() == value.NumElements(),
                errors::Internal("AssignVariableOp: element count mismatch after resize."));

    const int64_t n = value.NumElements();
    if (n == 0) return;

    auto stream = GetDeviceByCtx(ctx)->GetStream();
    const T* src = value.flat<T>().data();
    T* dst = var_tensor->flat<T>().data();

    LaunchAssignCopy<T>(src, dst, n, stream);
  }

 private:
  bool validate_shape_ = true;
};

// -------- 注册（TF2 only）--------
// resource 输入在 host；dtype 约束用 "dtype"
#define REGISTER_MUSA_ASSIGN_VAR(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("AssignVariableOp")                                             \
          .Device(DEVICE_MTGPU)                                            \
          .HostMemory("resource")                                          \
          .TypeConstraint<TYPE>("dtype"),                                  \
      MusaAssignVariableOp<TYPE>);

REGISTER_MUSA_ASSIGN_VAR(float);
REGISTER_MUSA_ASSIGN_VAR(double);
REGISTER_MUSA_ASSIGN_VAR(Eigen::half);
REGISTER_MUSA_ASSIGN_VAR(bfloat16);

#undef REGISTER_MUSA_ASSIGN_VAR

}  // namespace musa
}  // namespace tensorflow