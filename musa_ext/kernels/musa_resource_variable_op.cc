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
    *var->tensor() = value;
    var->is_initialized = true;
  }
};

class MusaReadVariableOp : public OpKernel {
 public:
  explicit MusaReadVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    const Tensor& handle_tensor = ctx->input(0);
    const ResourceHandle& handle = handle_tensor.flat<ResourceHandle>()(0);

    // std::cerr << ">>>>> [MUSA_READ_LOG] 2. Handle Name: " << handle.name() <<
    // ", Device: " << handle.device() << std::endl;

    Status s = LookupResource(ctx, handle, &var);
    if (!s.ok()) {
      //  std::cerr << ">>>>> [MUSA_READ_LOG] ❌ 3. LookupResource FAILED: " <<
      //  s.ToString() << std::endl;
      ctx->CtxFailure(s);
      return;
    }

    tf_shared_lock lock(*var->mu());

    if (!var->is_initialized) {
      //  std::cerr << ">>>>> [MUSA_READ_LOG] ❌ 4. Variable NOT Initialized!" <<
      //  std::endl;
      ctx->CtxFailure(errors::FailedPrecondition("Variable not initialized."));
      return;
    }

    const Tensor& t = *var->tensor();
    // std::cerr << ">>>>> [MUSA_READ_LOG] 5. Tensor Ready. DType: " <<
    // DataTypeString(t.dtype())
    //           << ", Shape: " << t.shape().DebugString() << std::endl;

    ctx->set_output(0, t);

    //  std::cerr << ">>>>> [MUSA_READ_LOG] 6. set_output(0) SUCCESS. Done." <<
    //  std::endl;
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
