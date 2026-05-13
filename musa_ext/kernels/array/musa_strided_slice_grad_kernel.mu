#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#define MAX_STRIDED_SLICE_GRAD_DIMS 8

struct StridedSliceGradLaunchParams {
  int rank;
  int use_64bit_index;
  int64_t inner_size;
  int64_t processing_shape[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t output_strides[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t begin[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t strides[MAX_STRIDED_SLICE_GRAD_DIMS];
};

template <typename Index>
struct StridedSliceGradIndexParams {
  int rank;
  Index inner_size;
  Index processing_shape[MAX_STRIDED_SLICE_GRAD_DIMS];
  Index output_strides[MAX_STRIDED_SLICE_GRAD_DIMS];
  Index begin[MAX_STRIDED_SLICE_GRAD_DIMS];
  Index strides[MAX_STRIDED_SLICE_GRAD_DIMS];
};

template <typename Index>
StridedSliceGradIndexParams<Index> MakeIndexParams(
    StridedSliceGradLaunchParams params) {
  StridedSliceGradIndexParams<Index> index_params;
  index_params.rank = params.rank;
  index_params.inner_size = static_cast<Index>(params.inner_size);
  for (int dim = 0; dim < MAX_STRIDED_SLICE_GRAD_DIMS; ++dim) {
    index_params.processing_shape[dim] =
        static_cast<Index>(params.processing_shape[dim]);
    index_params.output_strides[dim] =
        static_cast<Index>(params.output_strides[dim]);
    index_params.begin[dim] = static_cast<Index>(params.begin[dim]);
    index_params.strides[dim] = static_cast<Index>(params.strides[dim]);
  }
  return index_params;
}

template <typename T, typename Index>
__global__ void StridedSliceGradScatterKernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  Index index = tid;
  Index output_offset = 0;
  for (int dim = params.rank - 1; dim >= 0; --dim) {
    const Index dim_size = params.processing_shape[dim];
    const Index coord = dim_size == 0 ? 0 : index % dim_size;
    index = dim_size == 0 ? 0 : index / dim_size;
    const Index output_coord = params.begin[dim] + coord * params.strides[dim];
    output_offset += output_coord * params.output_strides[dim];
  }

  output[output_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradInnerContiguousKernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index inner_size = params.inner_size;
  Index index = tid / inner_size;
  const Index inner_offset = tid - index * inner_size;
  Index output_offset = inner_offset;
  for (int dim = params.rank - 1; dim >= 0; --dim) {
    const Index dim_size = params.processing_shape[dim];
    const Index coord = dim_size == 0 ? 0 : index % dim_size;
    index = dim_size == 0 ? 0 : index / dim_size;
    const Index output_coord = params.begin[dim] + coord * params.strides[dim];
    output_offset += output_coord * params.output_strides[dim];
  }

  output[output_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradInnerContiguousRank1Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index inner_size = params.inner_size;
  const Index c0 = tid / inner_size;
  const Index inner_offset = tid - c0 * inner_size;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  output[x0 * params.output_strides[0] + inner_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradInnerContiguousRank2Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index inner_size = params.inner_size;
  Index outer = tid / inner_size;
  const Index inner_offset = tid - outer * inner_size;
  const Index dim1 = params.processing_shape[1];
  const Index c1 = outer % dim1;
  const Index c0 = outer / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         inner_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradInnerContiguousRank3Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index inner_size = params.inner_size;
  Index outer = tid / inner_size;
  const Index inner_offset = tid - outer * inner_size;
  const Index dim2 = params.processing_shape[2];
  const Index dim1 = params.processing_shape[1];
  const Index c2 = outer % dim2;
  outer /= dim2;
  const Index c1 = outer % dim1;
  const Index c0 = outer / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  const Index x2 = params.begin[2] + c2 * params.strides[2];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2] + inner_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradInnerContiguousRank4Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index inner_size = params.inner_size;
  Index outer = tid / inner_size;
  const Index inner_offset = tid - outer * inner_size;
  const Index dim3 = params.processing_shape[3];
  const Index dim2 = params.processing_shape[2];
  const Index dim1 = params.processing_shape[1];
  const Index c3 = outer % dim3;
  outer /= dim3;
  const Index c2 = outer % dim2;
  outer /= dim2;
  const Index c1 = outer % dim1;
  const Index c0 = outer / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  const Index x2 = params.begin[2] + c2 * params.strides[2];
  const Index x3 = params.begin[3] + c3 * params.strides[3];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2] + x3 * params.output_strides[3] +
         inner_offset] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradRank1Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index x0 = params.begin[0] + tid * params.strides[0];
  output[x0 * params.output_strides[0]] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradRank2Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index dim1 = params.processing_shape[1];
  const Index c1 = tid % dim1;
  const Index c0 = tid / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1]] =
      dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradRank3Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index dim2 = params.processing_shape[2];
  const Index dim1 = params.processing_shape[1];
  const Index c2 = tid % dim2;
  const Index rem = tid / dim2;
  const Index c1 = rem % dim1;
  const Index c0 = rem / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  const Index x2 = params.begin[2] + c2 * params.strides[2];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2]] = dy[tid];
}

template <typename T, typename Index>
__global__ void StridedSliceGradRank4Kernel(
    const T* __restrict__ dy, T* __restrict__ output, Index total_elements,
    StridedSliceGradIndexParams<Index> params) {
  const Index tid =
      static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const Index dim3 = params.processing_shape[3];
  const Index dim2 = params.processing_shape[2];
  const Index dim1 = params.processing_shape[1];
  const Index c3 = tid % dim3;
  Index rem = tid / dim3;
  const Index c2 = rem % dim2;
  rem /= dim2;
  const Index c1 = rem % dim1;
  const Index c0 = rem / dim1;
  const Index x0 = params.begin[0] + c0 * params.strides[0];
  const Index x1 = params.begin[1] + c1 * params.strides[1];
  const Index x2 = params.begin[2] + c2 * params.strides[2];
  const Index x3 = params.begin[3] + c3 * params.strides[3];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2] + x3 * params.output_strides[3]] =
      dy[tid];
}

#define STRIDED_SLICE_GRAD_THREADS 256
#define STRIDED_SLICE_GRAD_BLOCKS(count) \
  (((count) + STRIDED_SLICE_GRAD_THREADS - 1) / STRIDED_SLICE_GRAD_THREADS)

template <typename T, typename Index>
void LaunchStridedSliceGradTypedIndex(
    const T* dy, T* output, Index total_elements,
    StridedSliceGradIndexParams<Index> params, musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = STRIDED_SLICE_GRAD_BLOCKS(total_elements);
  if (params.inner_size > 1) {
    if (params.rank == 1) {
      StridedSliceGradInnerContiguousRank1Kernel<T, Index>
          <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
              dy, output, total_elements, params);
    } else if (params.rank == 2) {
      StridedSliceGradInnerContiguousRank2Kernel<T, Index>
          <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
              dy, output, total_elements, params);
    } else if (params.rank == 3) {
      StridedSliceGradInnerContiguousRank3Kernel<T, Index>
          <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
              dy, output, total_elements, params);
    } else if (params.rank == 4) {
      StridedSliceGradInnerContiguousRank4Kernel<T, Index>
          <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
              dy, output, total_elements, params);
    } else {
      StridedSliceGradInnerContiguousKernel<T, Index>
          <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
              dy, output, total_elements, params);
    }
  } else if (params.rank == 1) {
    StridedSliceGradRank1Kernel<T, Index>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 2) {
    StridedSliceGradRank2Kernel<T, Index>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 3) {
    StridedSliceGradRank3Kernel<T, Index>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 4) {
    StridedSliceGradRank4Kernel<T, Index>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else {
    StridedSliceGradScatterKernel<T, Index>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  }
}

template <typename T>
void LaunchStridedSliceGradTyped(const T* dy, T* output,
                                 int64_t total_elements,
                                 StridedSliceGradLaunchParams params,
                                 musaStream_t stream) {
  if (params.use_64bit_index) {
    LaunchStridedSliceGradTypedIndex<T, int64_t>(
        dy, output, total_elements, MakeIndexParams<int64_t>(params), stream);
  } else {
    LaunchStridedSliceGradTypedIndex<T, int32_t>(
        dy, output, static_cast<int32_t>(total_elements),
        MakeIndexParams<int32_t>(params), stream);
  }
}

extern "C" {

#define DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(T, Name)                     \
  void Name(const T* dy, T* output, int64_t total_elements,             \
            StridedSliceGradLaunchParams params, musaStream_t stream) { \
    LaunchStridedSliceGradTyped<T>(dy, output, total_elements, params,  \
                                   stream);                             \
  }

DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(float, LaunchStridedSliceGradFloat)
DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(double, LaunchStridedSliceGradDouble)
DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(int32_t, LaunchStridedSliceGradInt32)
DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(int64_t, LaunchStridedSliceGradInt64)
DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(bool, LaunchStridedSliceGradBool)

void LaunchStridedSliceGradHalf(const void* dy, void* output,
                                int64_t total_elements,
                                StridedSliceGradLaunchParams params,
                                musaStream_t stream) {
  LaunchStridedSliceGradTyped<half>(
      reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(output),
      total_elements, params, stream);
}

void LaunchStridedSliceGradBFloat16(const void* dy, void* output,
                                    int64_t total_elements,
                                    StridedSliceGradLaunchParams params,
                                    musaStream_t stream) {
  LaunchStridedSliceGradTyped<__mt_bfloat16>(
      reinterpret_cast<const __mt_bfloat16*>(dy),
      reinterpret_cast<__mt_bfloat16*>(output), total_elements, params,
      stream);
}

#undef DEFINE_STRIDED_SLICE_GRAD_LAUNCHER
#undef STRIDED_SLICE_GRAD_BLOCKS
#undef STRIDED_SLICE_GRAD_THREADS

}  // extern "C"
