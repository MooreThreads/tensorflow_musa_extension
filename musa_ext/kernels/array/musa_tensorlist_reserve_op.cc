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

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(element_shape_tensor.shape()),
        errors::InvalidArgument("element_shape must be a vector, got shape ",
                                element_shape_tensor.shape().DebugString()));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(num_elements_tensor.shape()),
        errors::InvalidArgument("num_elements must be a scalar, got shape ",
                                num_elements_tensor.shape().DebugString()));

    PartialTensorShape element_shape;

    if (element_shape_tensor.NumElements() > 0) {
      OP_REQUIRES(ctx, element_shape_tensor.dtype() == DT_INT32,
                  errors::InvalidArgument(
                      "element_shape must be int32, got ",
                      DataTypeString(element_shape_tensor.dtype())));

      auto vec = element_shape_tensor.vec<int32>();
      std::vector<int64_t> dims(vec.size());
      for (int i = 0; i < vec.size(); ++i) {
        dims[i] = static_cast<int64_t>(vec(i));
      }

      OP_REQUIRES_OK(
          ctx, PartialTensorShape::MakePartialShape(
                   dims.data(), static_cast<int>(dims.size()), &element_shape));
    }

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