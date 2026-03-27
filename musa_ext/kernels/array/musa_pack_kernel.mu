#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// Unpack Batch Kernel
// Processes all outputs in a single kernel launch for better performance
// ============================================================================

template <typename T>
__global__ void UnpackBatchKernel(
    const T* __restrict__ input,
    T* const* __restrict__ outputs,
    int64_t outer_size,
    int64_t N,
    int64_t inner_size) {

  const int64_t total_elements = outer_size * N * inner_size;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  // Decompose position: [outer_size, N, inner_size]
  const int64_t inner_idx = tid % inner_size;
  const int64_t temp = tid / inner_size;
  const int64_t unpack_idx = temp % N;
  const int64_t outer_idx = temp / N;

  // Source offset in input
  const int64_t src_offset = (outer_idx * N + unpack_idx) * inner_size + inner_idx;

  // Destination offset in output[unpack_idx]: [outer_size, inner_size]
  const int64_t dst_offset = outer_idx * inner_size + inner_idx;

  outputs[unpack_idx][dst_offset] = input[src_offset];
}

// ============================================================================
// Unpack Single Output Kernel
// Writes to a single output tensor for a specific index
// ============================================================================

template <typename T>
__global__ void UnpackSingleKernel(
    const T* input,
    T* output,
    int64_t outer_size,
    int64_t N,
    int64_t inner_size,
    int64_t unpack_idx) {

  const int64_t output_size = outer_size * inner_size;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= output_size) return;

  // Decompose output position
  const int64_t inner_idx = tid % inner_size;
  const int64_t outer_idx = tid / inner_size;

  // Source offset in input: [outer_size, N, inner_size]
  const int64_t src_offset = (outer_idx * N + unpack_idx) * inner_size + inner_idx;

  output[tid] = input[src_offset];
}

// ============================================================================
// Launcher Functions
// ============================================================================

extern "C" {

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// ----------------------------------------------------------------------------
// Unpack Single Output Launchers
// ----------------------------------------------------------------------------

#define DEFINE_UNPACK_SINGLE_LAUNCHER(T, Name) \
  void Name(const T* input, T* output, \
            int64_t outer_size, int64_t N, int64_t inner_size, \
            int64_t unpack_idx, musaStream_t stream) { \
    const int64_t total_elements = outer_size * inner_size; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    UnpackSingleKernel<T><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        input, output, outer_size, N, inner_size, unpack_idx); \
  }

DEFINE_UNPACK_SINGLE_LAUNCHER(float, LaunchUnpackSingleFloat)
DEFINE_UNPACK_SINGLE_LAUNCHER(double, LaunchUnpackSingleDouble)
DEFINE_UNPACK_SINGLE_LAUNCHER(int32_t, LaunchUnpackSingleInt32)
DEFINE_UNPACK_SINGLE_LAUNCHER(int64_t, LaunchUnpackSingleInt64)
DEFINE_UNPACK_SINGLE_LAUNCHER(uint8_t, LaunchUnpackSingleUInt8)
DEFINE_UNPACK_SINGLE_LAUNCHER(bool, LaunchUnpackSingleBool)

#undef DEFINE_UNPACK_SINGLE_LAUNCHER

// ----------------------------------------------------------------------------
// Unpack Batch Output Launchers
// Processes all outputs in a single kernel for better performance
// ----------------------------------------------------------------------------

#define DEFINE_UNPACK_BATCH_LAUNCHER(T, Name) \
  void Name(const T* input, T* const* outputs, \
            int64_t outer_size, int64_t N, int64_t inner_size, \
            musaStream_t stream) { \
    const int64_t total_elements = outer_size * N * inner_size; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    UnpackBatchKernel<T><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        input, outputs, outer_size, N, inner_size); \
  }

DEFINE_UNPACK_BATCH_LAUNCHER(float, LaunchUnpackBatchFloat)
DEFINE_UNPACK_BATCH_LAUNCHER(double, LaunchUnpackBatchDouble)
DEFINE_UNPACK_BATCH_LAUNCHER(int32_t, LaunchUnpackBatchInt32)
DEFINE_UNPACK_BATCH_LAUNCHER(int64_t, LaunchUnpackBatchInt64)
DEFINE_UNPACK_BATCH_LAUNCHER(uint8_t, LaunchUnpackBatchUInt8)
DEFINE_UNPACK_BATCH_LAUNCHER(bool, LaunchUnpackBatchBool)

#undef DEFINE_UNPACK_BATCH_LAUNCHER

// Half precision - Single output
void LaunchUnpackSingleHalf(const void* input, void* output,
                            int64_t outer_size, int64_t N, int64_t inner_size,
                            int64_t unpack_idx, musaStream_t stream) {
  const int64_t total_elements = outer_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackSingleKernel<half><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(input),
      reinterpret_cast<half*>(output),
      outer_size, N, inner_size, unpack_idx);
}

// Half precision - Batch output
void LaunchUnpackBatchHalf(const void* input, void* const* outputs,
                           int64_t outer_size, int64_t N, int64_t inner_size,
                           musaStream_t stream) {
  const int64_t total_elements = outer_size * N * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackBatchKernel<half><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(input),
      reinterpret_cast<half* const*>(outputs),
      outer_size, N, inner_size);
}

// BFloat16 - Single output
void LaunchUnpackSingleBFloat16(const void* input, void* output,
                                int64_t outer_size, int64_t N, int64_t inner_size,
                                int64_t unpack_idx, musaStream_t stream) {
  const int64_t total_elements = outer_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackSingleKernel<__mt_bfloat16><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input),
      reinterpret_cast<__mt_bfloat16*>(output),
      outer_size, N, inner_size, unpack_idx);
}

// BFloat16 - Batch output
void LaunchUnpackBatchBFloat16(const void* input, void* const* outputs,
                               int64_t outer_size, int64_t N, int64_t inner_size,
                               musaStream_t stream) {
  const int64_t total_elements = outer_size * N * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackBatchKernel<__mt_bfloat16><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input),
      reinterpret_cast<__mt_bfloat16* const*>(outputs),
      outer_size, N, inner_size);
}

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"