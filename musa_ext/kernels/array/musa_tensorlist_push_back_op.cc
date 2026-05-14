// Implements TensorListPushBack for the MUSA device.
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
// TensorListPushBack
//
// Inputs:
//   0: input_handle  (HostMemory, Variant / TensorList)
//   1: tensor        (device memory, element to push)
// Outputs:
//   0: output_handle (HostMemory, Variant / TensorList)
// ============================================================
class MusaTensorListPushBackOp : public MusaOpKernel {
 public:
  explicit MusaTensorListPushBackOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_element = ctx->input(1);
    OP_REQUIRES(
        ctx, element_dtype_ == input_element.dtype(),
        errors::InvalidArgument("Invalid data types; list elements ",
                                DataTypeString(element_dtype_),
                                " but tried to append ",
                                DataTypeString(input_element.dtype())));

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

    OP_REQUIRES(
        ctx, l->element_shape.IsCompatibleWith(input_element.shape()),
        errors::InvalidArgument(
            "Tried to append a tensor with incompatible shape to a list. "
            "Op element shape: ",
            input_element.shape().DebugString(),
            " list shape: ", l->element_shape.DebugString()));

    if (l->max_num_elements != -1) {
      OP_REQUIRES(
          ctx,
          static_cast<int64_t>(l->tensors().size()) < l->max_num_elements,
          errors::InvalidArgument("Tried to push item into a full list",
                                  " list size: ", l->tensors().size(),
                                  " max_num_elements: ",
                                  l->max_num_elements));
    }

    Tensor* output_tensor = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor, attr));

    TensorList output_list = l->Copy();
    output_list.tensors().push_back(input_element);
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }

 private:
  DataType element_dtype_;
};

// TensorListPushBack: no TypeConstraint (handles all element_dtype values),
// matching the TF GPU registration pattern.
REGISTER_KERNEL_BUILDER(Name("TensorListPushBack")
                            .Device("MUSA")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle"),
                        MusaTensorListPushBackOp);

}  // namespace musa
}  // namespace tensorflow
