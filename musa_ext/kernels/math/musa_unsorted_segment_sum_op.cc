#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"

extern "C" {
void LaunchUnsortedSegmentSumFloatInt32(const float* data,
                                        const int* segment_ids,
                                        int num_segments, int64_t N, int64_t M,
                                        float* output, musaStream_t stream);
void LaunchUnsortedSegmentSumFloatInt64(const float* data,
                                        const int64_t* segment_ids,
                                        int64_t num_segments, int64_t N,
                                        int64_t M, float* output,
                                        musaStream_t stream);
void LaunchUnsortedSegmentSumDoubleInt32(const double* data,
                                         const int* segment_ids,
                                         int num_segments, int64_t N, int64_t M,
                                         double* output, musaStream_t stream);
void LaunchUnsortedSegmentSumDoubleInt64(const double* data,
                                         const int64_t* segment_ids,
                                         int64_t num_segments, int64_t N,
                                         int64_t M, double* output,
                                         musaStream_t stream);
void LaunchUnsortedSegmentSumInt32Int32(const int* data, const int* segment_ids,
                                        int num_segments, int64_t N, int64_t M,
                                        int* output, musaStream_t stream);
void LaunchUnsortedSegmentSumInt32Int64(const int* data,
                                        const int64_t* segment_ids,
                                        int64_t num_segments, int64_t N,
                                        int64_t M, int* output,
                                        musaStream_t stream);
void LaunchUnsortedSegmentSumInt64Int32(const int64_t* data,
                                        const int* segment_ids,
                                        int num_segments, int64_t N, int64_t M,
                                        int64_t* output, musaStream_t stream);
void LaunchUnsortedSegmentSumInt64Int64(const int64_t* data,
                                        const int64_t* segment_ids,
                                        int64_t num_segments, int64_t N,
                                        int64_t M, int64_t* output,
                                        musaStream_t stream);
// Half/BFloat16 via float accumulation
void LaunchUnsortedSegmentSumHalfToFloatInt32(const uint16_t* data,
                                               const int* segment_ids,
                                               int num_segments, int64_t N,
                                               int64_t M, float* float_output,
                                               musaStream_t stream);
void LaunchUnsortedSegmentSumHalfToFloatInt64(const uint16_t* data,
                                               const int64_t* segment_ids,
                                               int64_t num_segments, int64_t N,
                                               int64_t M, float* float_output,
                                               musaStream_t stream);
void LaunchUnsortedSegmentSumBFloat16ToFloatInt32(const uint16_t* data,
                                                   const int* segment_ids,
                                                   int num_segments, int64_t N,
                                                   int64_t M,
                                                   float* float_output,
                                                   musaStream_t stream);
void LaunchUnsortedSegmentSumBFloat16ToFloatInt64(const uint16_t* data,
                                                   const int64_t* segment_ids,
                                                   int64_t num_segments,
                                                   int64_t N, int64_t M,
                                                   float* float_output,
                                                   musaStream_t stream);
void LaunchConvertFloatToHalfBits(const float* input, uint16_t* output,
                                   int64_t n, musaStream_t stream);
void LaunchConvertFloatToBFloat16Bits(const float* input, uint16_t* output,
                                       int64_t n, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T, typename Tindex>
class UnsortedSegmentSumOp : public OpKernel {
 public:
  explicit UnsortedSegmentSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& segment_ids = ctx->input(1);
    const Tensor& num_segments_t = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_segments_t.shape()),
                errors::InvalidArgument("num_segments should be a scalar."));

    const Tindex num_segments = num_segments_t.scalar<Tindex>()();

    OP_REQUIRES(ctx, num_segments > 0,
                errors::InvalidArgument("num_segments should be positive."));

    const int64 N = segment_ids.NumElements();
    const int64 data_elements = data.NumElements();

    TensorShape output_shape;
    output_shape.AddDim(num_segments);
    TensorShape data_shape = data.shape();
    for (int i = segment_ids.dims(); i < data_shape.dims(); ++i) {
      output_shape.AddDim(data_shape.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    musaMemsetAsync(const_cast<char*>(output->tensor_data().data()), 0,
                    output->TotalBytes(), stream);

    if (N == 0 || data_elements == 0) return;

    OP_REQUIRES(ctx, data_elements % N == 0,
                errors::InvalidArgument(
                    "data size must be a multiple of segment_ids size."));

    const int64 M = data_elements / N;

    LaunchKernel(data.flat<T>().data(), segment_ids.flat<Tindex>().data(),
                 num_segments, N, M, output->flat<T>().data(), stream);
  }

 private:
  void LaunchKernel(const T* data, const Tindex* segment_ids,
                    Tindex num_segments, int64 N, int64 M, T* output,
                    musaStream_t stream);
};

#define SPECIALIZE_LAUNCHER(T, Tindex, FuncName)                              \
  template <>                                                                 \
  void UnsortedSegmentSumOp<T, Tindex>::LaunchKernel(                         \
      const T* data, const Tindex* segment_ids, Tindex num_segments, int64 N, \
      int64 M, T* output, musaStream_t stream) {                              \
    FuncName(data, segment_ids, num_segments, N, M, output, stream);          \
  }

SPECIALIZE_LAUNCHER(float, int32, LaunchUnsortedSegmentSumFloatInt32)
SPECIALIZE_LAUNCHER(float, int64, LaunchUnsortedSegmentSumFloatInt64)
SPECIALIZE_LAUNCHER(double, int32, LaunchUnsortedSegmentSumDoubleInt32)
SPECIALIZE_LAUNCHER(double, int64, LaunchUnsortedSegmentSumDoubleInt64)
SPECIALIZE_LAUNCHER(int32, int32, LaunchUnsortedSegmentSumInt32Int32)
SPECIALIZE_LAUNCHER(int32, int64, LaunchUnsortedSegmentSumInt32Int64)
SPECIALIZE_LAUNCHER(int64, int32, LaunchUnsortedSegmentSumInt64Int32)
SPECIALIZE_LAUNCHER(int64, int64, LaunchUnsortedSegmentSumInt64Int64)

#undef SPECIALIZE_LAUNCHER

#define REGISTER_MUSA_SEGMENT_SUM(type, index_type)                   \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                  \
                              .Device("MUSA")                         \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("num_segments"),            \
                          UnsortedSegmentSumOp<type, index_type>)

#define REGISTER_MUSA_SEGMENT_SUM_ALL(type) \
  REGISTER_MUSA_SEGMENT_SUM(type, int32);   \
  REGISTER_MUSA_SEGMENT_SUM(type, int64);

REGISTER_MUSA_SEGMENT_SUM_ALL(float);
REGISTER_MUSA_SEGMENT_SUM_ALL(double);
REGISTER_MUSA_SEGMENT_SUM_ALL(int32);
REGISTER_MUSA_SEGMENT_SUM_ALL(int64);

#undef REGISTER_MUSA_SEGMENT_SUM_ALL
#undef REGISTER_MUSA_SEGMENT_SUM

// ==========================================
// UnsortedSegmentSum for half/bfloat16
// via float accumulation (two-pass kernel)
// ==========================================
template <typename T, typename Tindex>
class UnsortedSegmentSumOpViaFloat : public OpKernel {
 public:
  explicit UnsortedSegmentSumOpViaFloat(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& segment_ids = ctx->input(1);
    const Tensor& num_segments_t = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_segments_t.shape()),
                errors::InvalidArgument("num_segments should be a scalar."));

    const Tindex num_segments = num_segments_t.scalar<Tindex>()();
    OP_REQUIRES(ctx, num_segments > 0,
                errors::InvalidArgument("num_segments should be positive."));

    const int64 N = segment_ids.NumElements();
    const int64 data_elements = data.NumElements();

    TensorShape output_shape;
    output_shape.AddDim(num_segments);
    TensorShape data_shape = data.shape();
    for (int i = segment_ids.dims(); i < data_shape.dims(); ++i) {
      output_shape.AddDim(data_shape.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    musaMemsetAsync(const_cast<char*>(output->tensor_data().data()), 0,
                    output->TotalBytes(), stream);

    if (N == 0 || data_elements == 0) return;

    OP_REQUIRES(ctx, data_elements % N == 0,
                errors::InvalidArgument(
                    "data size must be a multiple of segment_ids size."));

    const int64 M = data_elements / N;

    // Allocate float intermediate buffer
    Tensor float_buf;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, output_shape, &float_buf));
    musaMemsetAsync(float_buf.flat<float>().data(), 0,
                    float_buf.TotalBytes(), stream);

    const auto* data_bits =
        reinterpret_cast<const uint16_t*>(data.flat<T>().data());
    float* float_out = float_buf.flat<float>().data();
    auto* out_bits =
        reinterpret_cast<uint16_t*>(output->flat<T>().data());

    // Accumulate in float
    if (std::is_same<Tindex, int32>::value) {
      const int* seg = reinterpret_cast<const int*>(
          segment_ids.flat<Tindex>().data());
      if (std::is_same<T, Eigen::half>::value) {
        LaunchUnsortedSegmentSumHalfToFloatInt32(data_bits, seg, num_segments,
                                                  N, M, float_out, stream);
      } else {
        LaunchUnsortedSegmentSumBFloat16ToFloatInt32(data_bits, seg,
                                                      num_segments, N, M,
                                                      float_out, stream);
      }
    } else {
      const int64_t* seg = reinterpret_cast<const int64_t*>(
          segment_ids.flat<Tindex>().data());
      if (std::is_same<T, Eigen::half>::value) {
        LaunchUnsortedSegmentSumHalfToFloatInt64(data_bits, seg, num_segments,
                                                  N, M, float_out, stream);
      } else {
        LaunchUnsortedSegmentSumBFloat16ToFloatInt64(data_bits, seg,
                                                      num_segments, N, M,
                                                      float_out, stream);
      }
    }

    // Convert float -> T
    int64_t out_count = static_cast<int64_t>(num_segments) * M;
    if (std::is_same<T, Eigen::half>::value) {
      LaunchConvertFloatToHalfBits(float_out, out_bits, out_count, stream);
    } else {
      LaunchConvertFloatToBFloat16Bits(float_out, out_bits, out_count, stream);
    }
  }
};

#define REGISTER_MUSA_SEGMENT_SUM_VF(type, index_type)               \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                  \
                              .Device("MUSA")                         \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("num_segments"),            \
                          UnsortedSegmentSumOpViaFloat<type, index_type>)

REGISTER_MUSA_SEGMENT_SUM_VF(Eigen::half, int32);
REGISTER_MUSA_SEGMENT_SUM_VF(Eigen::half, int64);
REGISTER_MUSA_SEGMENT_SUM_VF(bfloat16, int32);
REGISTER_MUSA_SEGMENT_SUM_VF(bfloat16, int64);

#undef REGISTER_MUSA_SEGMENT_SUM_VF

}  // namespace musa
}  // namespace tensorflow
