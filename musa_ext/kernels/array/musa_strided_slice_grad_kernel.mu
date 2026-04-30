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

extern "C" {

#define STRIDED_SLICE_GRAD_THREADS 256
#define STRIDED_SLICE_GRAD_BLOCKS(count) \
  (((count) + STRIDED_SLICE_GRAD_THREADS - 1) / STRIDED_SLICE_GRAD_THREADS)

#define DEFINE_STRIDED_SLICE_GRAD_LAUNCHER(T, Name)                     \
  void Name(const T* dy, T* output, int64_t total_elements,             \
            StridedSliceGradLaunchParams params, musaStream_t stream) { \
    if (total_elements == 0) return;                                    \
    const int blocks = STRIDED_SLICE_GRAD_BLOCKS(total_elements);       \
    StridedSliceGradScatterKernel<T>                                    \
        <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(            \
            dy, output, total_elements, params);                        \
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
  if (total_elements == 0) return;
  const int blocks = STRIDED_SLICE_GRAD_BLOCKS(total_elements);
  StridedSliceGradScatterKernel<half>
      <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
          reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(output),
          total_elements, params);
}

void LaunchStridedSliceGradBFloat16(const void* dy, void* output,
                                    int64_t total_elements,
                                    StridedSliceGradLaunchParams params,
                                    musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = STRIDED_SLICE_GRAD_BLOCKS(total_elements);
  StridedSliceGradScatterKernel<__mt_bfloat16>
      <<<blocks, STRIDED_SLICE_GRAD_THREADS, 0, stream>>>(
          reinterpret_cast<const __mt_bfloat16*>(dy),
          reinterpret_cast<__mt_bfloat16*>(output), total_elements, params);
}

#undef DEFINE_STRIDED_SLICE_GRAD_LAUNCHER
#undef STRIDED_SLICE_GRAD_BLOCKS
#undef STRIDED_SLICE_GRAD_THREADS

}  // extern "C"
