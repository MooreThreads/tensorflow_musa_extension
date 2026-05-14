// Implements TensorListSetItem for the MUSA device.
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

// ============================================================
// TensorListSetItem
//
// Inputs:
//   0: input_handle  (HostMemory, Variant / TensorList)
//   1: index         (HostMemory, scalar int32)
//   2: item          (device memory, element to write)
// Outputs:
//   0: output_handle (HostMemory, Variant / TensorList with item replaced)
// ============================================================
class MusaTensorListSetItemOp : public MusaOpKernel {
 public:
  explicit MusaTensorListSetItemOp(OpKernelConstruction* ctx)
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
        errors::InvalidArgument("Trying to modify element ", index,
                                " in a list with ", l->tensors().size(),
                                " elements."));

    const Tensor& value = ctx->input(2);
    OP_REQUIRES(
        ctx, l->element_shape.IsCompatibleWith(value.shape()),
        errors::InvalidArgument(
            "Tried to set a tensor with incompatible shape at a list index. "
            "Item element shape: ",
            value.shape().DebugString(),
            " list shape: ", l->element_shape.DebugString()));

    Tensor* output_tensor = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor, attr));
    TensorList output_list = l->Copy();
    output_list.tensors()[index] = value;
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }

 private:
  DataType element_dtype_;
};

// TensorListSetItem: non-templated class registered per element_dtype.
#define REGISTER_MUSA_TENSOR_LIST_SET_ITEM(TYPE)                          \
  REGISTER_KERNEL_BUILDER(Name("TensorListSetItem")                       \
                              .Device("MUSA")                             \
                              .TypeConstraint<TYPE>("element_dtype")      \
                              .HostMemory("input_handle")                 \
                              .HostMemory("index")                        \
                              .HostMemory("output_handle"),               \
                          MusaTensorListSetItemOp)

REGISTER_MUSA_TENSOR_LIST_SET_ITEM(float);
REGISTER_MUSA_TENSOR_LIST_SET_ITEM(double);
REGISTER_MUSA_TENSOR_LIST_SET_ITEM(Eigen::half);
REGISTER_MUSA_TENSOR_LIST_SET_ITEM(bfloat16);
REGISTER_MUSA_TENSOR_LIST_SET_ITEM(int32);
REGISTER_MUSA_TENSOR_LIST_SET_ITEM(int64);
#undef REGISTER_MUSA_TENSOR_LIST_SET_ITEM

}  // namespace musa
}  // namespace tensorflow
