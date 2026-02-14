// musa_variable_v2_op.cc
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

using Var = ::tensorflow::Var;

template <typename T>
class MusaVariableV2Op : public OpKernel {
 public:
  explicit MusaVariableV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    const DataType dtype = DataTypeToEnum<T>::value;

    // Create or lookup the Var resource by (container, shared_name).
    Var* var = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<Var>(
                 container_, shared_name_, &var, [dtype](Var** ptr) -> Status {
                   *ptr = new Var(dtype);
                   return Status::OK();
                 }));

    core::ScopedUnref unref(var);

    // Make sure the backing tensor exists and has correct dtype/shape.
    {
      mutex_lock lock(*var->mu());

      // If first time, allocate storage tensor with required shape.
      if (var->tensor()->dtype() == DT_INVALID) {
        *var->tensor() = Tensor(dtype, shape_);
        var->is_initialized = false;  // VariableV2 itself does not initialize.
      } else {
        // Validate dtype/shape consistency if the variable already exists.
        OP_REQUIRES(
            ctx, var->tensor()->dtype() == dtype,
            errors::InvalidArgument("VariableV2 dtype mismatch. Existing: ",
                                    DataTypeString(var->tensor()->dtype()),
                                    ", requested: ", DataTypeString(dtype)));

        OP_REQUIRES(
            ctx, var->tensor()->shape() == shape_,
            errors::InvalidArgument("VariableV2 shape mismatch. Existing: ",
                                    var->tensor()->shape().DebugString(),
                                    ", requested: ", shape_.DebugString()));
      }
    }

    // IMPORTANT: VariableV2 output is a *ref* tensor.
    // Bind output(0) to the variable's backing tensor.
    ctx->set_output_ref(0, var->mu(), var->tensor());
  }

 private:
  string container_;
  string shared_name_;
  TensorShape shape_;
};

#define REGISTER_MUSA_VARIABLE_V2(T)                                \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("VariableV2").Device("MUSA").TypeConstraint<T>("dtype"), \
      MusaVariableV2Op<T>);

REGISTER_MUSA_VARIABLE_V2(float);
REGISTER_MUSA_VARIABLE_V2(double);
REGISTER_MUSA_VARIABLE_V2(Eigen::half);
REGISTER_MUSA_VARIABLE_V2(Eigen::bfloat16);
REGISTER_MUSA_VARIABLE_V2(int32);
REGISTER_MUSA_VARIABLE_V2(int64);

#undef REGISTER_MUSA_VARIABLE_V2

}  // namespace musa
}  // namespace tensorflow