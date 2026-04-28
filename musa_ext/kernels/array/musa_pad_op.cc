#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "../utils_op.h"

#ifndef MAX_DIM
#define MAX_DIM 4  // NHWC 4d support only
#endif

namespace tensorflow {
namespace musa {
extern "C" {
void nd_pad_kernel_launcher_float(const float *, float *, const int,
                                  const int64_t *, const int64_t *,
                                  const int64_t *, const int64_t *, const float,
                                  const int64_t, const musaStream_t);
void nd_pad_kernel_launcher_double(const double *, double *, const int,
                                   const int64_t *, const int64_t *,
                                   const int64_t *, const int64_t *,
                                   const double, const int64_t,
                                   const musaStream_t);
void nd_pad_kernel_launcher_int32(const int32_t *, int32_t *, const int,
                                  const int64_t *, const int64_t *,
                                  const int64_t *, const int64_t *,
                                  const int32_t, const int64_t,
                                  const musaStream_t);
void nd_pad_kernel_launcher_int64(const int64_t *, int64_t *, const int,
                                  const int64_t *, const int64_t *,
                                  const int64_t *, const int64_t *,
                                  const int64_t, const int64_t,
                                  const musaStream_t);
void nd_pad_kernel_launcher_uint8(const uint8_t *, uint8_t *, const int,
                                  const int64_t *, const int64_t *,
                                  const int64_t *, const int64_t *,
                                  const uint8_t, const int64_t,
                                  const musaStream_t);
}

namespace {
struct FloatTag {};
struct DoubleTag {};
struct Int32Tag {};
struct Int64Tag {};
struct Uint8Tag {};
struct UnsupportedTag {};

template <typename T>
struct TypeTag {
  using type = UnsupportedTag;
};

template <>
struct TypeTag<float> {
  using type = FloatTag;
};
template <>
struct TypeTag<double> {
  using type = DoubleTag;
};
template <>
struct TypeTag<int32_t> {
  using type = Int32Tag;
};
template <>
struct TypeTag<int64_t> {
  using type = Int64Tag;
};
template <>
struct TypeTag<uint8_t> {
  using type = Uint8Tag;
};

#define DEFINE_PAD_LAUNCHER_IMPL(T, TAG, SUFFIX)                              \
  void CallPadLauncherImpl(                                                   \
      const T *input_data, T *output_data, const int dims,                    \
      const int64_t *in_dims, const int64_t *out_dims,                        \
      const int64_t *pad_before, const int64_t *pad_after, const T pad_value, \
      const int64_t total_out_elements, const musaStream_t stream, TAG) {     \
    nd_pad_kernel_launcher_##SUFFIX(input_data, output_data, dims, in_dims,   \
                                    out_dims, pad_before, pad_after,          \
                                    pad_value, total_out_elements, stream);   \
  }

DEFINE_PAD_LAUNCHER_IMPL(float, FloatTag, float)
DEFINE_PAD_LAUNCHER_IMPL(double, DoubleTag, double)
DEFINE_PAD_LAUNCHER_IMPL(int32_t, Int32Tag, int32)
DEFINE_PAD_LAUNCHER_IMPL(int64_t, Int64Tag, int64)
DEFINE_PAD_LAUNCHER_IMPL(uint8_t, Uint8Tag, uint8)

#undef DEFINE_PAD_LAUNCHER_IMPL
void CallPadLauncherImpl(const void *, void *, const int, const int64_t *,
                         const int64_t *, const int64_t *, const int64_t *,
                         const int64_t, const int64_t, const musaStream_t,
                         UnsupportedTag) {
  throw std::invalid_argument("Pad not support current type");
}

template <typename T>
void CallPadLauncher(const T *input_data, T *output_data, const int dims,
                     const int64_t *in_dims, const int64_t *out_dims,
                     const int64_t *pad_before, const int64_t *pad_after,
                     const T pad_value, const int64_t total_out_elements,
                     const musaStream_t stream) {
  static_assert(!std::is_same<typename TypeTag<T>::type, UnsupportedTag>::value,
                "Unsupported type for nd_pad_kernel_launcher");

  CallPadLauncherImpl(input_data, output_data, dims, in_dims, out_dims,
                      pad_before, pad_after, pad_value, total_out_elements,
                      stream, typename TypeTag<T>::type());
}
}  // namespace

template <typename T, typename Tpadding>
class MusaPadOp : public OpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext *ctx) override {
    const Tensor &input = ctx->input(0);
    const Tensor &paddings = ctx->input(1);
    const int dims = input.dims();

    static constexpr int kMaxDims = 4;
    OP_REQUIRES(ctx, dims >= 1 && dims <= kMaxDims,
                errors::Unimplemented(
                    "MusaPadGeneric supports 1-4D input, got ", dims, "D"));
    OP_REQUIRES(
        ctx,
        paddings.dims() == 2 && paddings.dim_size(0) == dims &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("Paddings must be [", dims, ", 2], got ",
                                paddings.shape().DebugString()));

    T pad_value = T(0);
    if (ctx->num_inputs() == 3) {
      const Tensor &constant_values = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(constant_values.shape()),
                  errors::InvalidArgument("constant_values must be scalar"));
      pad_value = constant_values.scalar<T>()();
    }

    typename TTypes<Tpadding>::ConstMatrix paddings_mat =
        paddings.matrix<Tpadding>();
    int64_t pad_before[8], pad_after[8];
    for (int d = 0; d < dims; ++d) {
      pad_before[d] = static_cast<int64_t>(paddings_mat(d, 0));
      pad_after[d] = static_cast<int64_t>(paddings_mat(d, 1));
      OP_REQUIRES(ctx, pad_before[d] >= 0 && pad_after[d] >= 0,
                  errors::InvalidArgument("Paddings must be non-negative"));
    }

    TensorShape out_shape;
    int64_t out_size = 1;
    int64_t input_dims[8], out_dims[8];
    for (int d = 0; d < dims; ++d) {
      int64_t out_dim = input.dim_size(d) + pad_before[d] + pad_after[d];
      input_dims[d] = input.dim_size(d);
      out_dims[d] = out_dim;
      out_shape.AddDim(out_dim);
      out_size *= out_dim;
    }

    Tensor *output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (out_size == 0) return;

    Tensor pad_before_dev, pad_after_dev, pad_input_dims, pad_out_dims;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({dims}),
                                           &pad_before_dev));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT64, TensorShape({dims}), &pad_after_dev));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({dims}),
                                           &pad_input_dims));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT64, TensorShape({dims}), &pad_out_dims));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaMemcpyAsync(pad_before_dev.flat<int64_t>().data(), pad_before,
                    dims * sizeof(int64_t), musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(pad_after_dev.flat<int64_t>().data(), pad_after,
                    dims * sizeof(int64_t), musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(pad_input_dims.flat<int64_t>().data(), input_dims,
                    dims * sizeof(int64_t), musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(pad_out_dims.flat<int64_t>().data(), out_dims,
                    dims * sizeof(int64_t), musaMemcpyHostToDevice, stream);

    CallPadLauncher<T>(input.flat<T>().data(), output->flat<T>().data(), dims,
                       pad_input_dims.flat<int64_t>().data(),
                       pad_out_dims.flat<int64_t>().data(),
                       pad_before_dev.flat<int64_t>().data(),
                       pad_after_dev.flat<int64_t>().data(), pad_value,
                       out_size, stream);

    MusaDeviceContext *musa_device_context =
        static_cast<MusaDeviceContext *>(ctx->op_device_context());
    musa_device_context->ThenExecute(
        stream,
        [pad_before_dev, pad_after_dev, pad_input_dims, pad_out_dims]() {});

    musaError_t err = musaGetLastError();
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA Pad kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

#define REGISTER_MUSA_PAD_TYPE(TYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MusaPadOp<TYPE, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Pad")                             \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings"),            \
                          MusaPadOp<TYPE, int64>);                \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int32>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          MusaPadOp<TYPE, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                           \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T")          \
                              .TypeConstraint<int64>("Tpaddings") \
                              .HostMemory("paddings")             \
                              .HostMemory("constant_values"),     \
                          MusaPadOp<TYPE, int64>);

REGISTER_MUSA_PAD_TYPE(float);
REGISTER_MUSA_PAD_TYPE(int32);
REGISTER_MUSA_PAD_TYPE(int64);
REGISTER_MUSA_PAD_TYPE(double);
REGISTER_MUSA_PAD_TYPE(uint8);

#undef REGISTER_MUSA_PAD_TYPE
}  // namespace musa
}  // namespace tensorflow
