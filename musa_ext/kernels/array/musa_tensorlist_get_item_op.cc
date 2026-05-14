// Implements TensorListGetItem for the MUSA device.
// Reference: tensorflow/core/kernels/list_kernels.cc/.h

#include "../../utils/musa_tensor_list_utils.h"
#include "../utils_op.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

namespace {

// Parse a HostMemory shape tensor into a PartialTensorShape.
// Accepts a scalar value of -1 (meaning fully unknown shape), or a 1-D vector
// of dimension sizes.
Status TensorShapeFromTensorLocal(const Tensor& t, PartialTensorShape* out) {
  if (t.shape() == TensorShape({})) {
    if ((t.dtype() == DT_INT32 && t.scalar<int32_t>()() == -1) ||
        (t.dtype() == DT_INT64 && t.scalar<int64_t>()() == -1)) {
      *out = PartialTensorShape();  // fully unknown
      return Status::OK();
    }
    return errors::InvalidArgument(
        "The only valid scalar shape tensor is the fully unknown shape "
        "specified as -1.");
  }
  if (t.shape().dims() != 1) {
    return errors::InvalidArgument("Shape must be at most rank 1 but is rank ",
                                   t.shape().dims());
  }
  if (t.dtype() == DT_INT32) {
    return PartialTensorShape::MakePartialShape(t.vec<int32_t>().data(),
                                                t.NumElements(), out);
  }
  if (t.dtype() == DT_INT64) {
    return PartialTensorShape::MakePartialShape(t.vec<int64_t>().data(),
                                                t.NumElements(), out);
  }
  return errors::InvalidArgument(
      "Expected an int32 or int64 shape tensor; found ",
      DataTypeString(t.dtype()));
}

}  // namespace

// ============================================================
// TensorListGetItem
//
// Inputs:
//   0: input_handle  (HostMemory, Variant / TensorList)
//   1: index         (HostMemory, scalar int32)
//   2: element_shape (HostMemory, used when the slot is uninitialized)
// Outputs:
//   0: item          (device memory, the element at *index*)
// ============================================================
template <typename T>
class MusaTensorListGetItemOp : public MusaOpKernel {
 public:
  explicit MusaTensorListGetItemOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Variant& variant = ctx->input(0).scalar<Variant>()();
    const TensorList* l = variant.get<TensorList>();
    OP_REQUIRES(
        ctx, l != nullptr,
        errors::InvalidArgument("input_handle is not a valid TensorList."));

    OP_REQUIRES(
        ctx, element_dtype_ == l->element_dtype,
        errors::InvalidArgument("Invalid data types; op elements ",
                                DataTypeString(element_dtype_),
                                " but list elements ",
                                DataTypeString(l->element_dtype)));

    const int32 index = ctx->input(1).scalar<int32>()();
    OP_REQUIRES(
        ctx,
        index >= 0 && static_cast<size_t>(index) < l->tensors().size(),
        errors::InvalidArgument("Trying to access element ", index,
                                " in a list with ", l->tensors().size(),
                                " elements."));

    if (l->tensors()[index].dtype() != DT_INVALID) {
      // Element has been written; forward the device tensor directly.
      ctx->set_output(0, l->tensors()[index]);
    } else {
      // Element was reserved but never written; return a zero tensor.
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(ctx, TensorShapeFromTensorLocal(ctx->input(2),
                                                     &partial_element_shape));

      // Merge with the list's declared element shape.
      PartialTensorShape merged;
      OP_REQUIRES_OK(
          ctx, partial_element_shape.MergeWith(l->element_shape, &merged));
      partial_element_shape = merged;

      // Try to infer shape from other initialized list elements.
      if (!partial_element_shape.IsFullyDefined()) {
        for (const Tensor& t : l->tensors()) {
          if (t.dtype() != DT_INVALID) {
            PartialTensorShape tmp = partial_element_shape;
            OP_REQUIRES_OK(
                ctx, tmp.MergeWith(t.shape(), &partial_element_shape));
          }
        }
      }

      TensorShape element_shape;
      OP_REQUIRES(
          ctx, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument(
              "Trying to read an uninitialized tensor but element_shape is "
              "not fully defined: ",
              partial_element_shape.DebugString(),
              " and no list element is set."));

      Tensor* result;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, element_shape, &result));

      if (result->NumElements() > 0) {
        auto& h = GetHandleByCtx(ctx);
        musaStream_t stream =
            reinterpret_cast<musaStream_t>(h.GetStream());
        OP_REQUIRES(ctx, stream != nullptr,
                    errors::Internal("Failed to get valid MUSA stream."));
        auto err = musaMemsetAsync(result->flat<T>().data(), 0,
                                   result->TotalBytes(), stream);
        OP_REQUIRES(
            ctx, err == musaSuccess,
            errors::Internal(
                "musaMemsetAsync failed in TensorListGetItem, error code: ",
                static_cast<int>(err)));
      }
    }
  }

 private:
  DataType element_dtype_;
};

// TensorListGetItem: templated because zero-initialization requires T.
#define REGISTER_MUSA_TENSOR_LIST_GET_ITEM(TYPE)                          \
  REGISTER_KERNEL_BUILDER(Name("TensorListGetItem")                       \
                              .Device("MUSA")                             \
                              .TypeConstraint<TYPE>("element_dtype")      \
                              .HostMemory("input_handle")                 \
                              .HostMemory("index")                        \
                              .HostMemory("element_shape"),               \
                          MusaTensorListGetItemOp<TYPE>)

REGISTER_MUSA_TENSOR_LIST_GET_ITEM(float);
REGISTER_MUSA_TENSOR_LIST_GET_ITEM(double);
REGISTER_MUSA_TENSOR_LIST_GET_ITEM(Eigen::half);
REGISTER_MUSA_TENSOR_LIST_GET_ITEM(bfloat16);
REGISTER_MUSA_TENSOR_LIST_GET_ITEM(int32);
REGISTER_MUSA_TENSOR_LIST_GET_ITEM(int64);
#undef REGISTER_MUSA_TENSOR_LIST_GET_ITEM

}  // namespace musa
}  // namespace tensorflow
