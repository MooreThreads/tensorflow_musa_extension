#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kSmallPatchDepth = 128;
constexpr int kLinePatchDepth = 256;
constexpr int64_t kMinAnchorParallelism = 256;
constexpr int64_t kSmallTotalElements = 4096;

template <typename T>
__device__ __forceinline__ T ZeroValue() {
  return T();
}

template <>
__device__ __forceinline__ float ZeroValue<float>() {
  return 0.0f;
}

template <>
__device__ __forceinline__ Eigen::half ZeroValue<Eigen::half>() {
  Eigen::half result;
  __half raw = __float2half(0.0f);
  *reinterpret_cast<__half*>(&result) = raw;
  return result;
}

template <>
__device__ __forceinline__ bfloat16 ZeroValue<bfloat16>() {
  bfloat16 result;
  *reinterpret_cast<uint16_t*>(&result) = 0;
  return result;
}

inline int BlocksFor(int64_t work_items) {
  const int64_t blocks64 =
      (work_items + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(blocks64 > kMaxBlocks ? kMaxBlocks : blocks64);
}

inline bool ShouldUseAnchorCopyPath(int64_t anchors, int64_t patch_depth,
                                    int64_t total_elements,
                                    int64_t max_patch_depth) {
  return patch_depth <= max_patch_depth &&
         (anchors >= kMinAnchorParallelism ||
          total_elements <= kSmallTotalElements);
}

template <typename T>
__global__ void DirectCopyKernel(const T* __restrict__ input,
                                 T* __restrict__ output,
                                 int64_t total_elements) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t tid =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total_elements; tid += grid_stride) {
    output[tid] = input[tid];
  }
}

template <typename T>
__global__ void UnitPatchKernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t total_elements, int64_t in_rows,
                                int64_t in_cols, int64_t depth,
                                int64_t out_rows, int64_t out_cols,
                                int stride_rows, int stride_cols) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t tid =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total_elements; tid += grid_stride) {
    const int64_t channel = tid % depth;
    int64_t t = tid / depth;
    const int64_t out_col = t % out_cols;
    t /= out_cols;
    const int64_t out_row = t % out_rows;
    const int64_t batch = t / out_rows;

    const int64_t in_row = out_row * stride_rows;
    const int64_t in_col = out_col * stride_cols;
    const int64_t input_offset =
        ((batch * in_rows + in_row) * in_cols + in_col) * depth + channel;
    output[tid] = input[input_offset];
  }
}

template <typename T>
__global__ void RowPatchKernel(
    const T* __restrict__ input, T* __restrict__ output, int64_t anchors,
    int64_t in_rows, int64_t in_cols, int64_t depth, int64_t out_rows,
    int64_t out_cols, int ksize_rows, int stride_rows, int rate_rows,
    int pad_top) {
  const int64_t patch_depth = static_cast<int64_t>(ksize_rows) * depth;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t anchor =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       anchor < anchors; anchor += grid_stride) {
    const int64_t out_col = anchor % out_cols;
    int64_t t = anchor / out_cols;
    const int64_t out_row = t % out_rows;
    const int64_t batch = t / out_rows;
    const int64_t output_base = anchor * patch_depth;

    #pragma unroll 1
    for (int patch_row = 0; patch_row < ksize_rows; ++patch_row) {
      const int64_t in_row =
          out_row * stride_rows + patch_row * rate_rows - pad_top;
      const int64_t output_offset =
          output_base + static_cast<int64_t>(patch_row) * depth;
      if (in_row >= 0 && in_row < in_rows) {
        const int64_t input_offset =
            ((batch * in_rows + in_row) * in_cols + out_col) * depth;
        #pragma unroll 1
        for (int64_t channel = 0; channel < depth; ++channel) {
          output[output_offset + channel] = input[input_offset + channel];
        }
      } else {
        #pragma unroll 1
        for (int64_t channel = 0; channel < depth; ++channel) {
          output[output_offset + channel] = ZeroValue<T>();
        }
      }
    }
  }
}

template <typename T>
__global__ void ColPatchKernel(
    const T* __restrict__ input, T* __restrict__ output, int64_t anchors,
    int64_t in_rows, int64_t in_cols, int64_t depth, int64_t out_rows,
    int64_t out_cols, int ksize_cols, int stride_cols, int rate_cols,
    int pad_left) {
  const int64_t patch_depth = static_cast<int64_t>(ksize_cols) * depth;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t anchor =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       anchor < anchors; anchor += grid_stride) {
    const int64_t out_col = anchor % out_cols;
    int64_t t = anchor / out_cols;
    const int64_t out_row = t % out_rows;
    const int64_t batch = t / out_rows;
    const int64_t output_base = anchor * patch_depth;

    #pragma unroll 1
    for (int patch_col = 0; patch_col < ksize_cols; ++patch_col) {
      const int64_t in_col =
          out_col * stride_cols + patch_col * rate_cols - pad_left;
      const int64_t output_offset =
          output_base + static_cast<int64_t>(patch_col) * depth;
      if (in_col >= 0 && in_col < in_cols) {
        const int64_t input_offset =
            ((batch * in_rows + out_row) * in_cols + in_col) * depth;
        #pragma unroll 1
        for (int64_t channel = 0; channel < depth; ++channel) {
          output[output_offset + channel] = input[input_offset + channel];
        }
      } else {
        #pragma unroll 1
        for (int64_t channel = 0; channel < depth; ++channel) {
          output[output_offset + channel] = ZeroValue<T>();
        }
      }
    }
  }
}

template <typename T, bool CHECK_BOUNDS>
__global__ void AnchorPatchKernel(
    const T* __restrict__ input, T* __restrict__ output, int64_t anchors,
    int64_t in_rows, int64_t in_cols, int64_t depth, int64_t out_rows,
    int64_t out_cols, int ksize_rows, int ksize_cols, int stride_rows,
    int stride_cols, int rate_rows, int rate_cols, int pad_top, int pad_left) {
  const int64_t patch_depth =
      static_cast<int64_t>(ksize_rows) * ksize_cols * depth;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t anchor =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       anchor < anchors; anchor += grid_stride) {
    const int64_t out_col = anchor % out_cols;
    int64_t t = anchor / out_cols;
    const int64_t out_row = t % out_rows;
    const int64_t batch = t / out_rows;
    const int64_t output_base = anchor * patch_depth;

    #pragma unroll 1
    for (int patch_row = 0; patch_row < ksize_rows; ++patch_row) {
      const int64_t in_row =
          out_row * stride_rows + patch_row * rate_rows - pad_top;
      #pragma unroll 1
      for (int patch_col = 0; patch_col < ksize_cols; ++patch_col) {
        const int64_t in_col =
            out_col * stride_cols + patch_col * rate_cols - pad_left;
        const int64_t patch_base =
            (static_cast<int64_t>(patch_row) * ksize_cols + patch_col) * depth;
        const int64_t output_offset = output_base + patch_base;

        if (!CHECK_BOUNDS ||
            (in_row >= 0 && in_row < in_rows && in_col >= 0 &&
             in_col < in_cols)) {
          const int64_t input_offset =
              ((batch * in_rows + in_row) * in_cols + in_col) * depth;
          #pragma unroll 1
          for (int64_t channel = 0; channel < depth; ++channel) {
            output[output_offset + channel] = input[input_offset + channel];
          }
        } else {
          #pragma unroll 1
          for (int64_t channel = 0; channel < depth; ++channel) {
            output[output_offset + channel] = ZeroValue<T>();
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void ExtractImagePatchesKernel(
    const T* __restrict__ input, T* __restrict__ output,
    int64_t total_elements, int64_t in_rows, int64_t in_cols, int64_t depth,
    int64_t out_rows, int64_t out_cols, int ksize_rows, int ksize_cols,
    int stride_rows, int stride_cols, int rate_rows, int rate_cols, int pad_top,
    int pad_left) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t tid =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total_elements; tid += grid_stride) {
    const int64_t patch_depth =
        static_cast<int64_t>(ksize_rows) * ksize_cols * depth;
    const int64_t patch_channel = tid % depth;
    const int64_t patch_index = tid % patch_depth;
    const int64_t patch_col = (patch_index / depth) % ksize_cols;
    const int64_t patch_row = patch_index / (depth * ksize_cols);

    int64_t t = tid / patch_depth;
    const int64_t out_col = t % out_cols;
    t /= out_cols;
    const int64_t out_row = t % out_rows;
    const int64_t batch = t / out_rows;

    const int64_t in_row =
        out_row * stride_rows + patch_row * rate_rows - pad_top;
    const int64_t in_col =
        out_col * stride_cols + patch_col * rate_cols - pad_left;

    if (in_row >= 0 && in_row < in_rows && in_col >= 0 && in_col < in_cols) {
      const int64_t input_offset =
          ((batch * in_rows + in_row) * in_cols + in_col) * depth +
          patch_channel;
      output[tid] = input[input_offset];
    } else {
      output[tid] = ZeroValue<T>();
    }
  }
}

}  // namespace

template <typename T>
musaError_t LaunchExtractImagePatches(
    const T* input, T* output, int64_t total_elements, int64_t in_rows,
    int64_t in_cols, int64_t depth, int64_t out_rows, int64_t out_cols,
    int ksize_rows, int ksize_cols, int stride_rows, int stride_cols,
    int rate_rows, int rate_cols, int pad_top, int pad_left,
    musaStream_t stream) {
  if (total_elements <= 0) return musaSuccess;

  const int64_t patch_depth =
      static_cast<int64_t>(ksize_rows) * ksize_cols * depth;
  const int64_t anchors = total_elements / patch_depth;
  const bool unit_patch =
      ksize_rows == 1 && ksize_cols == 1 && rate_rows == 1 &&
      rate_cols == 1 && pad_top == 0 && pad_left == 0;

  if (unit_patch && stride_rows == 1 && stride_cols == 1 &&
      out_rows == in_rows && out_cols == in_cols) {
    DirectCopyKernel<T><<<BlocksFor(total_elements), kThreadsPerBlock, 0,
                          stream>>>(input, output, total_elements);
    return musaGetLastError();
  }

  if (unit_patch &&
      ShouldUseAnchorCopyPath(anchors, patch_depth, total_elements,
                              kSmallPatchDepth)) {
    AnchorPatchKernel<T, false>
        <<<BlocksFor(anchors), kThreadsPerBlock, 0, stream>>>(
            input, output, anchors, in_rows, in_cols, depth, out_rows, out_cols,
            ksize_rows, ksize_cols, stride_rows, stride_cols, rate_rows,
            rate_cols, pad_top, pad_left);
    return musaGetLastError();
  }

  if (unit_patch) {
    UnitPatchKernel<T><<<BlocksFor(total_elements), kThreadsPerBlock, 0,
                         stream>>>(input, output, total_elements, in_rows,
                                   in_cols, depth, out_rows, out_cols,
                                   stride_rows, stride_cols);
    return musaGetLastError();
  }

  if (ksize_cols == 1 && stride_cols == 1 && rate_cols == 1 &&
      pad_left == 0 &&
      ShouldUseAnchorCopyPath(anchors, patch_depth, total_elements,
                              kLinePatchDepth)) {
    RowPatchKernel<T><<<BlocksFor(anchors), kThreadsPerBlock, 0, stream>>>(
        input, output, anchors, in_rows, in_cols, depth, out_rows, out_cols,
        ksize_rows, stride_rows, rate_rows, pad_top);
    return musaGetLastError();
  }

  if (ksize_rows == 1 && stride_rows == 1 && rate_rows == 1 &&
      pad_top == 0 &&
      ShouldUseAnchorCopyPath(anchors, patch_depth, total_elements,
                              kLinePatchDepth)) {
    ColPatchKernel<T><<<BlocksFor(anchors), kThreadsPerBlock, 0, stream>>>(
        input, output, anchors, in_rows, in_cols, depth, out_rows, out_cols,
        ksize_cols, stride_cols, rate_cols, pad_left);
    return musaGetLastError();
  }

  if (ShouldUseAnchorCopyPath(anchors, patch_depth, total_elements,
                              kSmallPatchDepth)) {
    const bool no_bounds =
        pad_top == 0 && pad_left == 0 &&
        (out_rows - 1) * stride_rows +
                static_cast<int64_t>(ksize_rows - 1) * rate_rows <
            in_rows &&
        (out_cols - 1) * stride_cols +
                static_cast<int64_t>(ksize_cols - 1) * rate_cols <
            in_cols;
    if (no_bounds) {
      AnchorPatchKernel<T, false>
          <<<BlocksFor(anchors), kThreadsPerBlock, 0, stream>>>(
              input, output, anchors, in_rows, in_cols, depth, out_rows,
              out_cols, ksize_rows, ksize_cols, stride_rows, stride_cols,
              rate_rows, rate_cols, pad_top, pad_left);
    } else {
      AnchorPatchKernel<T, true>
          <<<BlocksFor(anchors), kThreadsPerBlock, 0, stream>>>(
              input, output, anchors, in_rows, in_cols, depth, out_rows,
              out_cols, ksize_rows, ksize_cols, stride_rows, stride_cols,
              rate_rows, rate_cols, pad_top, pad_left);
    }
    return musaGetLastError();
  }

  const int blocks = BlocksFor(total_elements);
  ExtractImagePatchesKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, output, total_elements, in_rows, in_cols, depth, out_rows, out_cols,
      ksize_rows, ksize_cols, stride_rows, stride_cols, rate_rows, rate_cols,
      pad_top, pad_left);

  return musaGetLastError();
}

template musaError_t LaunchExtractImagePatches<float>(
    const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int, int, musaStream_t);
template musaError_t LaunchExtractImagePatches<double>(
    const double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int, int, musaStream_t);
template musaError_t LaunchExtractImagePatches<int32>(
    const int32*, int32*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int, int, musaStream_t);
template musaError_t LaunchExtractImagePatches<int64>(
    const int64*, int64*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int, int, musaStream_t);
template musaError_t LaunchExtractImagePatches<Eigen::half>(
    const Eigen::half*, Eigen::half*, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int, int, int, int, int, int, int, int, musaStream_t);
template musaError_t LaunchExtractImagePatches<bfloat16>(
    const bfloat16*, bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int, int, int, int, int, int, int, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
