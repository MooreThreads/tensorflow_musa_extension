#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/lib/core/notification.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_device.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

using Var = ::tensorflow::Var;

Status CopyTensorWithDeviceContext(OpKernelContext* ctx, const Tensor& src,
                                   Tensor* dst) {
  if (src.TotalBytes() == 0) {
    return Status::OK();
  }

  auto* device_context = ctx->op_device_context();
  if (device_context == nullptr) {
    // Fall back to direct MUSA memcpy if device context is not available
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    MusaMemcpyAsyncD2D(const_cast<char*>(dst->tensor_data().data()),
                       src.tensor_data().data(), src.TotalBytes(), stream);
    return Status::OK();
  }

  Device* device = static_cast<Device*>(ctx->device());
  Notification n;
  Status status;
  device_context->CopyTensorInSameDevice(&src, device, dst,
                                         [&n, &status](const Status& s) {
                                           status = s;
                                           n.Notify();
                                         });
  n.WaitForNotification();
  return status;
}

class MusaVarHandleOp : public OpKernel {
 public:
  explicit MusaVarHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_and_shape_.dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &dtype_and_shape_.shape));

    // Match TensorFlow's resource-variable behavior: if shared_name is empty,
    // each VarHandleOp instance must get its own unique resource name.
    if (shared_name_.empty()) {
      shared_name_ = ctx->def().name();
    }

    is_anonymous_ = shared_name_ == ResourceHandle::ANONYMOUS_NAME;

    if (!is_anonymous_) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                             &resource_, attr));
      resource_.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
          ctx, container_, shared_name_,
          std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_});
    }
  }

  // VarHandleOp is a lightweight metadata operation
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    if (is_anonymous_) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      Tensor handle;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
      handle.scalar<ResourceHandle>()() = MakeResourceHandle<Var>(
          ctx, container_, shared_name_,
          std::vector<DtypeAndPartialTensorShape>{dtype_and_shape_},
          ctx->stack_trace());
      ctx->set_output(0, handle);
    } else {
      ctx->set_output(0, resource_);
    }
  }

 private:
  string container_;
  string shared_name_;
  DtypeAndPartialTensorShape dtype_and_shape_;
  bool is_anonymous_;
  Tensor resource_;
};

template <typename T>
class MusaAssignVariableOp : public OpKernel {
 public:
  explicit MusaAssignVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  // AssignVariableOp is a lightweight operation (just pointer/reference
  // passing)
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(1);
    OP_REQUIRES(
        ctx, dtype_ == value.dtype(),
        errors::InvalidArgument(
            "Variable and value dtypes don't match; respectively, ",
            DataTypeString(dtype_), " and ", DataTypeString(value.dtype())));

    if (ctx->num_outputs() > 0) {
      ctx->set_output(0, ctx->input(0));
    }

    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(
                            ctx, HandleFromInput(ctx, 0), &var, [&](Var** ptr) {
                              *ptr = new Var(dtype_);
                              *(*ptr)->tensor() = value;
                              (*ptr)->is_initialized = true;
                              return Status::OK();
                            }));

    mutex_lock lock(*var->mu());

    OP_REQUIRES(
        ctx,
        (var->tensor()->dtype() == DT_INVALID && !var->is_initialized) ||
            var->tensor()->dtype() == dtype_,
        errors::InvalidArgument(
            "Trying to assign variable with wrong dtype. Expected ",
            DataTypeString(var->tensor()->dtype()), " got ",
            DataTypeString(dtype_)));

    if (var->copy_on_read_mode.load()) {
      AllocatorAttributes alloc_attr;
      alloc_attr.set_gpu_compatible(true);
      alloc_attr.set_nic_compatible(true);

      Tensor copied_value;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(value.dtype(), value.shape(),
                                             &copied_value, alloc_attr));
      OP_REQUIRES_OK(ctx,
                     CopyTensorWithDeviceContext(ctx, value, &copied_value));

      *var->tensor() = copied_value;
    } else {
      *var->tensor() = value;
    }

    var->is_initialized = true;
  }

 private:
  DataType dtype_;
};

class MusaReadVariableOp : public OpKernel {
 public:
  explicit MusaReadVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  // ReadVariableOp is a zero-copy metadata operation
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    const Tensor& handle_tensor = ctx->input(0);
    const ResourceHandle& handle = handle_tensor.flat<ResourceHandle>()(0);

    Status s = LookupResource(ctx, handle, &var);
    if (!s.ok()) {
      ctx->CtxFailure(s);
      return;
    }

    tf_shared_lock lock(*var->mu());

    if (!var->is_initialized) {
      ctx->CtxFailure(errors::FailedPrecondition("Variable not initialized."));
      return;
    }

    const Tensor& t = *var->tensor();
    OP_REQUIRES(ctx, dtype_ == t.dtype(),
                errors::InvalidArgument(
                    "Trying to read variable with wrong dtype. Expected ",
                    DataTypeString(dtype_), " got ",
                    DataTypeString(t.dtype())));

    // Always copy the tensor to ensure the returned tensor is independent
    // of the variable's underlying storage. This is critical for correct
    // eager execution semantics where read_value() should return the value
    // at the time of reading, not a reference that changes when the variable
    // is modified.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, t.shape(), &out));

    if (t.TotalBytes() > 0) {
      // Use direct MUSA memcpy instead of device context
      musaStream_t stream = GetMusaStreamByCtx(ctx);
      MusaMemcpyAsyncD2D(const_cast<char*>(out->tensor_data().data()),
                         t.tensor_data().data(), t.TotalBytes(), stream);
    }
  }

 private:
  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(
    Name("ReadVariableOp").Device("MUSA").HostMemory("resource"),
    MusaReadVariableOp);

REGISTER_KERNEL_BUILDER(
    Name("ResourceReadVariableOp").Device("MUSA").HostMemory("resource"),
    MusaReadVariableOp);

class MusaVarIsInitializedOp : public OpKernel {
 public:
  explicit MusaVarIsInitializedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // VarIsInitializedOp is a lightweight check operation
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    core::RefCountPtr<Var> var;
    bool is_init = LookupResource(ctx, HandleFromInput(ctx, 0), &var).ok() &&
                   var->is_initialized;
    out->flat<bool>()(0) = is_init;
  }
};

class MusaDestroyResourceOp : public OpKernel {
 public:
  explicit MusaDestroyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("ignore_lookup_error", &ignore_lookup_error_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    Status status = DeleteResource(ctx, HandleFromInput(ctx, 0));
    if (ignore_lookup_error_ && errors::IsNotFound(status)) {
      return;
    }
    OP_REQUIRES_OK(ctx, status);
  }

 private:
  bool ignore_lookup_error_;
};

#define REGISTER_MUSA_VAR_MANAGEMENT(T)                    \
  REGISTER_KERNEL_BUILDER(Name("VarHandleOp")              \
                              .Device("MUSA")              \
                              .HostMemory("resource")      \
                              .TypeConstraint<T>("dtype"), \
                          MusaVarHandleOp);                \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")         \
                              .Device("MUSA")              \
                              .HostMemory("resource")      \
                              .TypeConstraint<T>("dtype"), \
                          MusaAssignVariableOp<T>);

REGISTER_MUSA_VAR_MANAGEMENT(float);
REGISTER_MUSA_VAR_MANAGEMENT(double);
REGISTER_MUSA_VAR_MANAGEMENT(Eigen::half);
REGISTER_MUSA_VAR_MANAGEMENT(int32);
REGISTER_MUSA_VAR_MANAGEMENT(int64);

REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp")
                            .Device("MUSA")
                            .HostMemory("resource")
                            .HostMemory("is_initialized"),
                        MusaVarIsInitializedOp);
REGISTER_KERNEL_BUILDER(
    Name("DestroyResourceOp").Device("MUSA").HostMemory("resource"),
    MusaDestroyResourceOp);

}  // namespace musa
}  // namespace tensorflow
