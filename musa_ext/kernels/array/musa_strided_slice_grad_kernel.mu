#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#define MAX_STRIDED_SLICE_GRAD_DIMS 8

struct StridedSliceGradLaunchParams {
  int rank;
  int64_t processing_shape[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t output_strides[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t begin[MAX_STRIDED_SLICE_GRAD_DIMS];
  int64_t strides[MAX_STRIDED_SLICE_GRAD_DIMS];
};

template <typename T>
__global__ void StridedSliceGradScatterKernel(
    const T* __restrict__ dy, T* __restrict__ output, int64_t total_elements,
    StridedSliceGradLaunchParams params) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  int64_t index = tid;
  int64_t output_offset = 0;
  for (int dim = params.rank - 1; dim >= 0; --dim) {
    const int64_t dim_size = params.processing_shape[dim];
    const int64_t coord = dim_size == 0 ? 0 : index % dim_size;
    index = dim_size == 0 ? 0 : index / dim_size;
    const int64_t output_coord =
        params.begin[dim] + coord * params.strides[dim];
    output_offset += output_coord * params.output_strides[dim];
  }

  output[output_offset] = dy[tid];
}

template <typename T>
__global__ void StridedSliceGradRank1Kernel(
    const T* __restrict__ dy, T* __restrict__ output, int64_t total_elements,
    StridedSliceGradLaunchParams params) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const int64_t x0 = params.begin[0] + tid * params.strides[0];
  output[x0 * params.output_strides[0]] = dy[tid];
}

template <typename T>
__global__ void StridedSliceGradRank2Kernel(
    const T* __restrict__ dy, T* __restrict__ output, int64_t total_elements,
    StridedSliceGradLaunchParams params) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const int64_t dim1 = params.processing_shape[1];
  const int64_t c1 = tid % dim1;
  const int64_t c0 = tid / dim1;
  const int64_t x0 = params.begin[0] + c0 * params.strides[0];
  const int64_t x1 = params.begin[1] + c1 * params.strides[1];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1]] =
      dy[tid];
}

template <typename T>
__global__ void StridedSliceGradRank3Kernel(
    const T* __restrict__ dy, T* __restrict__ output, int64_t total_elements,
    StridedSliceGradLaunchParams params) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const int64_t dim2 = params.processing_shape[2];
  const int64_t dim1 = params.processing_shape[1];
  const int64_t c2 = tid % dim2;
  const int64_t rem = tid / dim2;
  const int64_t c1 = rem % dim1;
  const int64_t c0 = rem / dim1;
  const int64_t x0 = params.begin[0] + c0 * params.strides[0];
  const int64_t x1 = params.begin[1] + c1 * params.strides[1];
  const int64_t x2 = params.begin[2] + c2 * params.strides[2];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2]] = dy[tid];
}

template <typename T>
__global__ void StridedSliceGradRank4Kernel(
    const T* __restrict__ dy, T* __restrict__ output, int64_t total_elements,
    StridedSliceGradLaunchParams params) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  const int64_t dim3 = params.processing_shape[3];
  const int64_t dim2 = params.processing_shape[2];
  const int64_t dim1 = params.processing_shape[1];
  const int64_t c3 = tid % dim3;
  int64_t rem = tid / dim3;
  const int64_t c2 = rem % dim2;
  rem /= dim2;
  const int64_t c1 = rem % dim1;
  const int64_t c0 = rem / dim1;
  const int64_t x0 = params.begin[0] + c0 * params.strides[0];
  const int64_t x1 = params.begin[1] + c1 * params.strides[1];
  const int64_t x2 = params.begin[2] + c2 * params.strides[2];
  const int64_t x3 = params.begin[3] + c3 * params.strides[3];
  output[x0 * params.output_strides[0] + x1 * params.output_strides[1] +
         x2 * params.output_strides[2] + x3 * params.output_strides[3]] =
      dy[tid];
}

#define STRIDED_SLICE_GRAD_THREADS 256
#define STRIDED_SLICE_GRAD_BLOCKS(count) \
  (((count) + STRIDED_SLICE_GRAD_THREADS - 1) / STRIDED_SLICE_GRAD_THREADS)

template <typename T>
void LaunchStridedSliceGradTyped(const T* dy, T* output,
                                 int64_t total_elements,
                                 StridedSliceGradLaunchParams params,
                                 musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = STRIDED_SLICE_GRAD_BLOCKS(total_elements);
  if (params.rank == 1) {
    StridedSliceGradRank1Kernel<T>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 2) {
    StridedSliceGradRank2Kernel<T>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 3) {
    StridedSliceGradRank3Kernel<T>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else if (params.rank == 4) {
    StridedSliceGradRank4Kernel<T>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
  } else {
    StridedSliceGradScatterKernel<T>
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
            dy, output, total_elements, params);
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
