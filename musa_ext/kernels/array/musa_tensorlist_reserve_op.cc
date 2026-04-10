#include "../../utils/musa_tensor_list_utils.h"
#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

// Helper function to convert tensor to PartialTensorShape, handling both
// vector and scalar (-1 for unknown shape) inputs, same as EmptyTensorList.
Status TensorShapeFromTensorReserve(const Tensor& t, PartialTensorShape* out) {
  if (t.shape() == TensorShape({})) {
    // Scalar case: only valid if value is -1 (unknown shape)
    if ((t.dtype() == DT_INT32 && t.scalar<int32_t>()() == -1) ||
        (t.dtype() == DT_INT64 && t.scalar<int64_t>()() == -1)) {
      *out = PartialTensorShape();  // Fully unknown shape
      return ::tsl::OkStatus();
    }
    return errors::InvalidArgument(
        "The only valid scalar shape tensor is the fully unknown shape "
        "specified as -1.");
  } else if (t.shape().dims() != 1) {
    return errors::InvalidArgument("Shape must be at most rank 1 but is rank ",
                                   t.shape().dims());
  }
  // Vector case: standard path
  if (t.dtype() == DT_INT32) {
    return PartialTensorShape::MakePartialShape(t.vec<int32_t>().data(),
                                                t.NumElements(), out);
  } else if (t.dtype() == DT_INT64) {
    return PartialTensorShape::MakePartialShape(t.vec<int64_t>().data(),
                                                t.NumElements(), out);
  }
  return errors::InvalidArgument(
      "Expected an int32 or int64 shape tensor; found ",
      DataTypeString(t.dtype()));
}

class MusaTensorListReserveOp : public MusaOpKernel {
 public:
  explicit MusaTensorListReserveOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& element_shape_tensor = ctx->input(0);
    const Tensor& num_elements_tensor = ctx->input(1);

    // Handle both scalar (-1 for unknown shape) and vector element_shape
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(ctx, TensorShapeFromTensorReserve(element_shape_tensor, &element_shape));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(num_elements_tensor.shape()),
        errors::InvalidArgument("num_elements must be a scalar, got shape ",
                                num_elements_tensor.shape().DebugString()));

    int64_t num_elements = 0;
    if (num_elements_tensor.dtype() == DT_INT32) {
      num_elements =
          static_cast<int64_t>(num_elements_tensor.scalar<int32>()());
    } else if (num_elements_tensor.dtype() == DT_INT64) {
      num_elements = num_elements_tensor.scalar<int64_t>()();
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("num_elements must be int32 or int64, got ",
                                  DataTypeString(num_elements_tensor.dtype())));
    }

    OP_REQUIRES(ctx, num_elements >= 0,
                errors::InvalidArgument(
                    "num_elements must be non-negative, got ", num_elements));

    Tensor* output_handle = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &output_handle));

    TensorList output_list;
    output_list.element_dtype = element_dtype_;
    output_list.element_shape = element_shape;

    output_list.tensors().resize(num_elements);

    output_handle->scalar<Variant>()() = std::move(output_list);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListReserve")
                            .Device("MUSA")
                            .HostMemory("element_shape")
                            .HostMemory("num_elements")
                            .HostMemory("handle"),
                        MusaTensorListReserveOp);

}  // namespace musa
}  // namespace tensorflow