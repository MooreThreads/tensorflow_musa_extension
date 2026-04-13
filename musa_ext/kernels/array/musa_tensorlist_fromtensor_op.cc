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

Status TensorShapeFromTensorMusa(const Tensor& t, PartialTensorShape* out) {
  if (TensorShapeUtils::IsScalar(t.shape())) {
    if (t.dtype() == DT_INT32) {
      out->Clear();
      out->AddDim(static_cast<int64_t>(t.scalar<int32>()()));
      return ::tsl::OkStatus();
    } else if (t.dtype() == DT_INT64) {
      out->Clear();
      out->AddDim(static_cast<int64_t>(t.scalar<int64>()()));
      return ::tsl::OkStatus();
    } else {
      return errors::InvalidArgument(
          "element_shape must be int32 or int64, got ",
          DataTypeString(t.dtype()));
    }
  }

  if (!TensorShapeUtils::IsVector(t.shape())) {
    return errors::InvalidArgument(
        "element_shape must be a scalar or vector, got shape ",
        t.shape().DebugString());
  }

  out->Clear();

  if (t.dtype() == DT_INT32) {
    auto vec = t.vec<int32>();
    for (int i = 0; i < vec.size(); ++i) {
      out->AddDim(static_cast<int64_t>(vec(i)));
    }
    return ::tsl::OkStatus();
  }

  if (t.dtype() == DT_INT64) {
    auto vec = t.vec<int64>();
    for (int i = 0; i < vec.size(); ++i) {
      out->AddDim(static_cast<int64_t>(vec(i)));
    }
    return ::tsl::OkStatus();
  }

  return errors::InvalidArgument("element_shape must be int32 or int64, got ",
                                 DataTypeString(t.dtype()));
}

}  // namespace

template <typename T>
class MusaTensorListFromTensorOp : public MusaOpKernel {
 public:
  explicit MusaTensorListFromTensorOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor, attr));

    const Tensor& input_tensor = ctx->input(0);
    const Tensor& element_shape_tensor = ctx->input(1);

    OP_REQUIRES(
        ctx, !TensorShapeUtils::IsMatrixOrHigher(element_shape_tensor.shape()),
        errors::InvalidArgument(
            "TensorListFromTensor: element_shape must be at most rank 1 but "
            "has shape ",
            element_shape_tensor.shape().DebugString()));

    PartialTensorShape element_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeFromTensorMusa(element_shape_tensor, &element_shape));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(input_tensor.shape()),
                errors::InvalidArgument(
                    "Tensor must be at least a vector, but saw shape: ",
                    input_tensor.shape().DebugString()));

    TensorShape expected_element_shape(input_tensor.shape());
    expected_element_shape.RemoveDim(0);

    OP_REQUIRES(ctx, element_shape.IsCompatibleWith(expected_element_shape),
                errors::InvalidArgument("Specified a list with shape ",
                                        element_shape.DebugString(),
                                        " from a tensor with shape ",
                                        expected_element_shape.DebugString()));

    TensorList output_list;
    output_list.element_dtype = input_tensor.dtype();
    output_list.element_shape = element_shape;
    output_list.tensors().reserve(input_tensor.shape().dim_size(0));

    auto& h = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(h.GetStream());

    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("Failed to get valid MUSA stream."));

    const int64_t num_slices = input_tensor.shape().dim_size(0);

    for (int64_t i = 0; i < num_slices; ++i) {
      Tensor tmp = input_tensor.SubSlice(i);

      Tensor aligned;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(tmp.dtype(), tmp.shape(), &aligned));

      const size_t bytes =
          static_cast<size_t>(tmp.NumElements()) * DataTypeSize(tmp.dtype());

      if (bytes > 0) {
        void* dst = tensorflow::DMAHelper::base(&aligned);
        const void* src = tensorflow::DMAHelper::base(&tmp);

        OP_REQUIRES(ctx, dst != nullptr,
                    errors::Internal("DMAHelper::base(&aligned) is nullptr"));
        OP_REQUIRES(ctx, src != nullptr,
                    errors::Internal("DMAHelper::base(&tmp) is nullptr"));

        auto err =
            musaMemcpyAsync(dst, src, bytes, musaMemcpyDeviceToDevice, stream);

        OP_REQUIRES(
            ctx, err == musaSuccess,
            errors::Internal(
                "musaMemcpyAsync failed in TensorListFromTensor, error code: ",
                static_cast<int>(err)));
      }

      output_list.tensors().push_back(aligned);
    }

    output_tensor->scalar<Variant>()() = std::move(output_list);
  }
};

#define REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListFromTensor")                \
                              .Device("MUSA")                         \
                              .HostMemory("element_shape")            \
                              .HostMemory("output_handle")            \
                              .TypeConstraint<TYPE>("element_dtype"), \
                          MusaTensorListFromTensorOp<TYPE>)

REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(float);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(double);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(Eigen::half);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(bfloat16);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(int32);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(int64);
REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR(uint8);

#undef REGISTER_MUSA_TENSOR_LIST_FROM_TENSOR

}  // namespace musa
}  // namespace tensorflow
