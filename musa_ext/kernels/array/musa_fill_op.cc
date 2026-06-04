#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"

#include <algorithm>
#include <musa_runtime.h>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace musa {

namespace {

extern "C" {
void LaunchMusaFill_float(float* out, float value, int64_t n,
                          musaStream_t stream);
void LaunchMusaFill_double(double* out, double value, int64_t n,
                           musaStream_t stream);
void LaunchMusaFill_int32(int32* out, int32 value, int64_t n,
                          musaStream_t stream);
void LaunchMusaFill_int64(int64* out, int64 value, int64_t n,
                          musaStream_t stream);
void LaunchMusaFill_half(Eigen::half* out, Eigen::half value, int64_t n,
                         musaStream_t stream);
void LaunchMusaFill_bfloat16(Eigen::bfloat16* out, Eigen::bfloat16 value,
                             int64_t n, musaStream_t stream);
void LaunchMusaFill_bool(bool* out, bool value, int64_t n, musaStream_t stream);
}

template <typename T, typename... Rest>
struct is_any : std::false_type {};

template <typename T, typename First>
struct is_any<T, First> : std::is_same<T, First> {};

template <typename T, typename First, typename... Rest>
struct is_any<T, First, Rest...>
    : std::integral_constant<bool, std::is_same<T, First>::value ||
                                       is_any<T, Rest...>::value> {};

template <typename T>
Status LaunchFillKernel(T* out, T value, int64_t n, musaStream_t stream);

#define DEFINE_FILL_LAUNCHER(T, suffix)                                      \
  template <>                                                                \
  Status LaunchFillKernel<T>(T* out, T value, int64_t n, musaStream_t stream) { \
    LaunchMusaFill_##suffix(out, value, n, stream);                          \
    musaError_t err = musaGetLastError();                                    \
    if (err != musaSuccess) {                                                \
      return errors::Internal("MUSA Fill kernel launch failed: ",            \
                              musaGetErrorString(err));                      \
    }                                                                        \
    return OkStatus();                                                       \
  }

DEFINE_FILL_LAUNCHER(float, float)
DEFINE_FILL_LAUNCHER(double, double)
DEFINE_FILL_LAUNCHER(int32, int32)
DEFINE_FILL_LAUNCHER(int64, int64)
DEFINE_FILL_LAUNCHER(Eigen::half, half)
DEFINE_FILL_LAUNCHER(Eigen::bfloat16, bfloat16)
DEFINE_FILL_LAUNCHER(bool, bool)

#undef DEFINE_FILL_LAUNCHER

bool IsDeviceReadablePointer(const void* ptr) {
  musaPointerAttributes attributes;
  musaError_t attr_err = musaPointerGetAttributes(&attributes, ptr);
  if (attr_err == musaSuccess) {
    return attributes.type == musaMemoryTypeDevice ||
           attributes.type == musaMemoryTypeManaged;
  }

  // Pageable host memory is reported as unregistered on some MUSA versions.
  musaGetLastError();
  return false;
}

template <typename T>
Status ReadVectorInputToHost(OpKernelContext* context, int input_index,
                             std::vector<T>* values) {
  const Tensor& tensor = context->input(input_index);
  values->resize(tensor.NumElements());
  if (values->empty()) {
    return OkStatus();
  }

  const T* input_data = tensor.flat<T>().data();
  if (IsDeviceReadablePointer(input_data)) {
    mStatus copy_status =
        MusaMemcpyD2H(values->data(), input_data, tensor.TotalBytes());
    if (copy_status != mStatus::SUCCESS) {
      return errors::Internal("MUSA Fill failed to copy input ", input_index,
                              " to host.");
    }
  } else {
    std::copy(input_data, input_data + values->size(), values->data());
  }
  return OkStatus();
}

template <typename T>
Status ReadScalarInputToHost(OpKernelContext* context, int input_index,
                             T* value) {
  const Tensor& tensor = context->input(input_index);
  if (tensor.NumElements() < 1) {
    return errors::InvalidArgument("MUSA Fill input ", input_index,
                                   " must contain a scalar value.");
  }

  const T* input_data = tensor.flat<T>().data();
  if (IsDeviceReadablePointer(input_data)) {
    mStatus copy_status = MusaMemcpyD2H(value, input_data, sizeof(T));
    if (copy_status != mStatus::SUCCESS) {
      return errors::Internal("MUSA Fill failed to copy scalar input ",
                              input_index, " to host.");
    }
  } else {
    *value = input_data[0];
  }
  return OkStatus();
}

}  // namespace

template <typename T, typename Index>
class MusaFillOp : public MusaOpKernel {
 public:
  explicit MusaFillOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  // Fill is memory-intensive but not computationally expensive
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    const Tensor& Tdims = context->input(0);
    const Tensor& Tvalue = context->input(1);

    OP_REQUIRES(
        context,
        (TensorShapeUtils::IsVector(Tdims.shape()) ||
         TensorShapeUtils::IsScalar(Tdims.shape())),
        errors::InvalidArgument("dims must represent a vector, got shape ",
                                Tdims.shape().DebugString()));

    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(Tvalue.shape()) ||
            (TensorShapeUtils::IsVector(Tvalue.shape()) &&
             Tvalue.shape().dim_size(0) == 1),
        errors::InvalidArgument("value must represent a scalar, got shape ",
                                Tvalue.shape().DebugString()));

    std::vector<Index> host_dims;
    OP_REQUIRES_OK(context, ReadVectorInputToHost<Index>(context, 0, &host_dims));

    TensorShape shape;
    OP_REQUIRES_OK(
        context,
        TensorShapeUtils::MakeShape(host_dims.data(), host_dims.size(), &shape));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));

    if (shape.num_elements() == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(context);
    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("MUSA stream is unavailable for Fill"));

    T host_value;
    OP_REQUIRES_OK(context, ReadScalarInputToHost<T>(context, 1, &host_value));

    OP_REQUIRES_OK(context,
                   LaunchFillKernel<T>(out->flat<T>().data(), host_value,
                                       shape.num_elements(), stream));
  }
};

#define REGISTER_FILL_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device("MUSA")                        \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int32>("index_type")   \
                              .HostMemory("dims")                    \
                              .HostMemory("value"),                  \
                          MusaFillOp<type, int32>);                  \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device("MUSA")                        \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int64_t>("index_type") \
                              .HostMemory("dims")                    \
                              .HostMemory("value"),                  \
                          MusaFillOp<type, int64>);

REGISTER_FILL_KERNEL(float);
REGISTER_FILL_KERNEL(double);
REGISTER_FILL_KERNEL(int32);
REGISTER_FILL_KERNEL(int64);
REGISTER_FILL_KERNEL(Eigen::half);
REGISTER_FILL_KERNEL(Eigen::bfloat16);
REGISTER_FILL_KERNEL(bool);

#undef REGISTER_FILL_KERNEL

}  // namespace musa
}  // namespace tensorflow
