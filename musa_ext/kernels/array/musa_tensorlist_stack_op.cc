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

template <typename T>
class MusaTensorListStackOp : public MusaOpKernel {
 public:
  explicit MusaTensorListStackOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_elements", &num_elements_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_handle = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(input_handle.shape()),
        errors::InvalidArgument("input_handle must be a scalar, got shape ",
                                input_handle.shape().DebugString()));

    const Variant& variant = input_handle.scalar<Variant>()();
    const TensorList* tensor_list = variant.get<TensorList>();

    OP_REQUIRES(ctx, tensor_list != nullptr,
                errors::InvalidArgument(
                    "input_handle does not contain a valid TensorList."));

    OP_REQUIRES(
        ctx, tensor_list->element_dtype == element_dtype_,
        errors::InvalidArgument(
            "Invalid data types: op expects ", DataTypeString(element_dtype_),
            " but list contains ", DataTypeString(tensor_list->element_dtype)));

    if (num_elements_ != -1) {
      OP_REQUIRES(
          ctx, static_cast<int>(tensor_list->tensors().size()) == num_elements_,
          errors::InvalidArgument("Operation expected a list with ",
                                  num_elements_, " elements but got ",
                                  tensor_list->tensors().size(), " elements."));
    }

    const Tensor& element_shape_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(element_shape_tensor.shape()),
        errors::InvalidArgument("element_shape must be a vector, got shape ",
                                element_shape_tensor.shape().DebugString()));

    PartialTensorShape partial_element_shape;

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

      OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                              dims.data(), static_cast<int>(dims.size()),
                              &partial_element_shape));
    }

    if (partial_element_shape.dims() == -1 &&
        tensor_list->element_shape.dims() != -1) {
      partial_element_shape = tensor_list->element_shape;
    } else if (partial_element_shape.dims() != -1 &&
               tensor_list->element_shape.dims() != -1) {
      PartialTensorShape merged_shape;
      OP_REQUIRES_OK(ctx, partial_element_shape.MergeWith(
                              tensor_list->element_shape, &merged_shape));
      partial_element_shape = merged_shape;
    }

    for (const auto& t : tensor_list->tensors()) {
      if (t.dtype() != DT_INVALID) {
        if (partial_element_shape.dims() == -1) {
          partial_element_shape = t.shape();
        } else {
          PartialTensorShape merged_shape;
          OP_REQUIRES_OK(
              ctx, partial_element_shape.MergeWith(t.shape(), &merged_shape));
          partial_element_shape = merged_shape;
        }
      }
    }

    OP_REQUIRES(
        ctx,
        partial_element_shape.IsFullyDefined() ||
            !tensor_list->tensors().empty(),
        errors::InvalidArgument(
            "Tried to stack elements of an empty list with non-fully-defined "
            "element_shape: ",
            partial_element_shape.DebugString()));

    TensorShape element_shape;
    OP_REQUIRES(
        ctx, partial_element_shape.AsTensorShape(&element_shape),
        errors::InvalidArgument(
            "TensorListStack requires fully defined element shape, got ",
            partial_element_shape.DebugString()));

    TensorShape output_shape = element_shape;
    output_shape.InsertDim(0, tensor_list->tensors().size());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

    auto& h = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(h.GetStream());

    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("Failed to get valid MUSA stream."));

    Tensor zeros;
    bool has_zeros = false;

    T* output_base = output->flat<T>().data();
    int64_t offset_elems = 0;

    for (const auto& t : tensor_list->tensors()) {
      const Tensor* src_tensor = nullptr;

      if (t.dtype() != DT_INVALID) {
        src_tensor = &t;
      } else {
        if (!has_zeros) {
          OP_REQUIRES_OK(
              ctx, ctx->allocate_temp(element_dtype_, element_shape, &zeros));

          auto err = musaMemsetAsync(zeros.flat<T>().data(), 0,
                                     zeros.TotalBytes(), stream);
          OP_REQUIRES(
              ctx, err == musaSuccess,
              errors::Internal("musaMemsetAsync failed when initializing "
                               "temporary zero tensor, error code: ",
                               static_cast<int>(err)));
          has_zeros = true;
        }
        src_tensor = &zeros;
      }

      OP_REQUIRES(ctx, src_tensor->shape().IsSameSize(element_shape),
                  errors::InvalidArgument(
                      "TensorListStack expects each element to have "
                      "shape ",
                      element_shape.DebugString(), " but got ",
                      src_tensor->shape().DebugString()));

      const T* src_base = src_tensor->flat<T>().data();
      const size_t bytes = src_tensor->TotalBytes();

      auto err = musaMemcpyAsync(output_base + offset_elems, src_base, bytes,
                                 musaMemcpyHostToDevice, stream);
      OP_REQUIRES(
          ctx, err == musaSuccess,
          errors::Internal("musaMemcpyAsync failed in TensorListStack, error "
                           "code: ",
                           static_cast<int>(err)));

      offset_elems += src_tensor->NumElements();
    }
  }

 private:
  int num_elements_;
  DataType element_dtype_;
};

#define REGISTER_MUSA_TENSOR_LIST_STACK(TYPE)                         \
  REGISTER_KERNEL_BUILDER(Name("TensorListStack")                     \
                              .Device("MUSA")                         \
                              .HostMemory("input_handle")             \
                              .HostMemory("element_shape")            \
                              .TypeConstraint<TYPE>("element_dtype"), \
                          MusaTensorListStackOp<TYPE>)

REGISTER_MUSA_TENSOR_LIST_STACK(float);
REGISTER_MUSA_TENSOR_LIST_STACK(double);
REGISTER_MUSA_TENSOR_LIST_STACK(Eigen::half);
REGISTER_MUSA_TENSOR_LIST_STACK(bfloat16);
REGISTER_MUSA_TENSOR_LIST_STACK(int32);
REGISTER_MUSA_TENSOR_LIST_STACK(int64);
REGISTER_MUSA_TENSOR_LIST_STACK(uint8);
REGISTER_MUSA_TENSOR_LIST_STACK(bool);

#undef REGISTER_MUSA_TENSOR_LIST_STACK

}  // namespace musa
}  // namespace tensorflow