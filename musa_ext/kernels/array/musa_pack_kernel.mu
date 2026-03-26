// MUSA Pack/Unpack Custom Kernels
// Optimized kernel implementation for Pack and Unpack operations
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// Optimized Pack Kernel - one thread handles all N positions on axis
// Better memory coalescing and reduced pointer array access
// ============================================================================

template <typename T>
__global__ void PackKernelOptimized(
    const T* const* inputs,
    T* output,
    int64_t outer_size,
    int64_t N,
    int64_t inner_size) {

  // Each thread handles all N elements for one (outer_idx, inner_idx) position
  const int64_t per_thread_size = outer_size * inner_size;
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= per_thread_size) return;

  // Decompose into outer_idx and inner_idx
  const int64_t inner_idx = tid % inner_size;
  const int64_t outer_idx = tid / inner_size;

  // Source offset in each input tensor
  const int64_t src_offset = outer_idx * inner_size + inner_idx;

  // Output base offset for this (outer_idx, inner_idx) position
  // Output layout: [outer_size, N, inner_size]
  const int64_t out_base = (outer_idx * N) * inner_size + inner_idx;

  // Process all N inputs
  for (int64_t i = 0; i < N; ++i) {
    output[out_base + i * inner_size] = inputs[i][src_offset];
  }
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
// Pack Launchers
// ----------------------------------------------------------------------------

#define DEFINE_PACK_LAUNCHER(T, Name) \
  void Name(const T* const* inputs, T* output, \
            int64_t outer_size, int64_t N, int64_t inner_size, \
            musaStream_t stream) { \
    const int64_t total_elements = outer_size * N * inner_size; \
    if (total_elements == 0) return; \
    const int64_t per_thread_size = outer_size * inner_size; \
    const int blocks = OPTIMAL_BLOCKS(per_thread_size); \
    PackKernelOptimized<T><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        inputs, output, outer_size, N, inner_size); \
  }

DEFINE_PACK_LAUNCHER(float, LaunchPackFloat)
DEFINE_PACK_LAUNCHER(double, LaunchPackDouble)
DEFINE_PACK_LAUNCHER(int32_t, LaunchPackInt32)
DEFINE_PACK_LAUNCHER(int64_t, LaunchPackInt64)
DEFINE_PACK_LAUNCHER(uint8_t, LaunchPackUInt8)
DEFINE_PACK_LAUNCHER(bool, LaunchPackBool)

#undef DEFINE_PACK_LAUNCHER

// Half precision
void LaunchPackHalf(const void* const* inputs, void* output,
                    int64_t outer_size, int64_t N, int64_t inner_size,
                    musaStream_t stream) {
  const int64_t total_elements = outer_size * N * inner_size;
  if (total_elements == 0) return;
  const int64_t per_thread_size = outer_size * inner_size;
  const int blocks = OPTIMAL_BLOCKS(per_thread_size);
  PackKernelOptimized<half><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half* const*>(inputs),
      reinterpret_cast<half*>(output),
      outer_size, N, inner_size);
}

// BFloat16
void LaunchPackBFloat16(const void* const* inputs, void* output,
                        int64_t outer_size, int64_t N, int64_t inner_size,
                        musaStream_t stream) {
  const int64_t total_elements = outer_size * N * inner_size;
  if (total_elements == 0) return;
  const int64_t per_thread_size = outer_size * inner_size;
  const int blocks = OPTIMAL_BLOCKS(per_thread_size);
  PackKernelOptimized<__mt_bfloat16><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16* const*>(inputs),
      reinterpret_cast<__mt_bfloat16*>(output),
      outer_size, N, inner_size);
}

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

// Half precision
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

// BFloat16
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

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"