// Implements TensorListPopBack for the MUSA device.
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
// TensorListPopBack
//
// Inputs:
//   0: input_handle  (HostMemory, Variant / TensorList)
//   1: element_shape (HostMemory, used when the back element is uninitialized)
// Outputs:
//   0: output_handle (HostMemory, Variant / TensorList with last entry removed)
//   1: tensor        (device memory, the popped element)
// ============================================================
template <typename T>
class MusaTensorListPopBackOp : public MusaOpKernel {
 public:
  explicit MusaTensorListPopBackOp(OpKernelConstruction* ctx)
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

    OP_REQUIRES(ctx, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));

    OP_REQUIRES(ctx, !l->tensors().empty(),
                errors::InvalidArgument("Trying to pop from an empty list."));

    const Tensor& t = l->tensors().back();
    if (t.dtype() != DT_INVALID) {
      // The element is already initialized; pass the device tensor directly.
      ctx->set_output(1, t);
    } else {
      // Element was reserved but never written; allocate a zero tensor.
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(ctx, TensorShapeFromTensorLocal(ctx->input(1),
                                                     &partial_element_shape));
      // Merge with the list's declared element shape.
      PartialTensorShape merged;
      OP_REQUIRES_OK(
          ctx, partial_element_shape.MergeWith(l->element_shape, &merged));
      partial_element_shape = merged;

      TensorShape element_shape;
      OP_REQUIRES(
          ctx, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument(
              "Trying to read an uninitialized tensor but element_shape is "
              "not fully defined: ",
              partial_element_shape.DebugString()));

      Tensor* result;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, element_shape, &result));

      if (result->NumElements() > 0) {
        auto& h = GetHandleByCtx(ctx);
        musaStream_t stream = reinterpret_cast<musaStream_t>(h.GetStream());
        OP_REQUIRES(ctx, stream != nullptr,
                    errors::Internal("Failed to get valid MUSA stream."));
        auto err = musaMemsetAsync(result->flat<T>().data(), 0,
                                   result->TotalBytes(), stream);
        OP_REQUIRES(
            ctx, err == musaSuccess,
            errors::Internal(
                "musaMemsetAsync failed in TensorListPopBack, error code: ",
                static_cast<int>(err)));
      }
    }

    // Build the output list with the last element removed.
    Tensor* output_handle = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_handle, attr));
    TensorList output_list = l->Copy();
    output_list.tensors().pop_back();
    output_handle->scalar<Variant>()() = std::move(output_list);
  }

 private:
  DataType element_dtype_;
};

// TensorListPopBack: templated because zero-initialization requires T.
#define REGISTER_MUSA_TENSOR_LIST_POP_BACK(TYPE)                     \
  REGISTER_KERNEL_BUILDER(Name("TensorListPopBack")                  \
                              .Device("MUSA")                        \
                              .TypeConstraint<TYPE>("element_dtype") \
                              .HostMemory("input_handle")            \
                              .HostMemory("element_shape")           \
                              .HostMemory("output_handle"),          \
                          MusaTensorListPopBackOp<TYPE>)

REGISTER_MUSA_TENSOR_LIST_POP_BACK(float);
REGISTER_MUSA_TENSOR_LIST_POP_BACK(double);
REGISTER_MUSA_TENSOR_LIST_POP_BACK(Eigen::half);
REGISTER_MUSA_TENSOR_LIST_POP_BACK(bfloat16);
REGISTER_MUSA_TENSOR_LIST_POP_BACK(int32);
REGISTER_MUSA_TENSOR_LIST_POP_BACK(int64);
#undef REGISTER_MUSA_TENSOR_LIST_POP_BACK

}  // namespace musa
}  // namespace tensorflow
