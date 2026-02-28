#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

using Var = ::tensorflow::Var;

class MusaVarHandleOp : public OpKernel {
 public:
  explicit MusaVarHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  }

  // VarHandleOp is a lightweight metadata operation
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    ResourceHandle handle =
        MakeResourceHandle<Var>(ctx, container_, shared_name_);
    out->flat<ResourceHandle>()(0) = handle;
  }

 private:
  string container_;
  string shared_name_;
};

template <typename T>
class MusaAssignVariableOp : public OpKernel {
 public:
  explicit MusaAssignVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // AssignVariableOp is a lightweight operation (just pointer/reference passing)
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(1);

    if (ctx->num_outputs() > 0) {
      ctx->set_output(0, ctx->input(0));
    }

    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(
                            ctx, HandleFromInput(ctx, 0), &var, [&](Var** ptr) {
                              *ptr = new Var(value.dtype());
                              return Status::OK();
                            }));

    mutex_lock lock(*var->mu());

    // Use CopyFrom instead of operator= for better performance
    // CopyFrom will forward the buffer if possible (ref count == 1),
    // otherwise it will perform a proper device-to-device copy
    OP_REQUIRES(
        ctx,
        var->tensor()->CopyFrom(value, value.shape()),
        errors::Internal("Failed to assign value to variable. Expected shape: ",
                         var->tensor()->shape().DebugString(),
                         ", got shape: ", value.shape().DebugString()));

    var->is_initialized = true;
  }
};

class MusaReadVariableOp : public OpKernel {
 public:
  explicit MusaReadVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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

    // OPTIMIZATION: set_output creates a shallow copy (alias) of the tensor
    // This is zero-copy - it only increments the reference count of the buffer
    // The actual data remains in GPU memory without any memcpy
    ctx->set_output(0, t);
  }
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
  explicit MusaDestroyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    DeleteResource(ctx, HandleFromInput(ctx, 0));
  }
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
