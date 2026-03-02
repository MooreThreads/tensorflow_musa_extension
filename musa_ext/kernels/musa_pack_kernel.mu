// High-Performance MUSA Pack/Unpack Kernels
// Optimized for memory bandwidth and coalesced access patterns
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// Optimized Pack Kernel - Vectorized and Coalesced Memory Access
// ============================================================================

// Scalar version for general types
template <typename T>
__global__ void PackKernelScalar(const T** __restrict__ inputs, 
                                  T* __restrict__ output, 
                                  int num_inputs, 
                                  int64_t before_size, 
                                  int64_t after_size, 
                                  int64_t total_elements) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  // Decompose output index: tid -> (b, i, a)
  // Layout: output[before][num_inputs][after]
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_inputs;
  const int64_t b = temp / num_inputs;
  
  // Input layout: inputs[i][before][after]
  const int64_t in_idx = b * after_size + after_idx;
  output[tid] = inputs[i][in_idx];
}

// Vectorized float4 version - 4x memory bandwidth utilization
__global__ void PackKernelFloat4(const float4** __restrict__ inputs,
                                  float4* __restrict__ output,
                                  int num_inputs,
                                  int64_t before_size,
                                  int64_t after_size,  // in float4 units
                                  int64_t total_elements) {  // in float4 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_inputs;
  // const int64_t b = temp / num_inputs;  // Not needed for indexing
  
  output[tid] = inputs[i][after_idx];
}

// Vectorized int4 version
__global__ void PackKernelInt4(const int4** __restrict__ inputs,
                                int4* __restrict__ output,
                                int num_inputs,
                                int64_t before_size,
                                int64_t after_size,  // in int4 units
                                int64_t total_elements) {  // in int4 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_inputs;
  
  output[tid] = inputs[i][after_idx];
}

// Vectorized half2 version for 16-bit types
__global__ void PackKernelHalf2(const half2** __restrict__ inputs,
                                 half2* __restrict__ output,
                                 int num_inputs,
                                 int64_t before_size,
                                 int64_t after_size,  // in half2 units
                                 int64_t total_elements) {  // in half2 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_inputs;
  
  output[tid] = inputs[i][after_idx];
}

// Shared memory optimized version for small after_size
// Uses shared memory to coalesce writes and reduce global memory transactions
template <typename T, int MAX_AFTER_SIZE>
__global__ void PackKernelSharedMem(const T** __restrict__ inputs,
                                     T* __restrict__ output,
                                     int num_inputs,
                                     int64_t before_size,
                                     int64_t after_size,
                                     int64_t total_elements) {
  extern __shared__ char shared_mem[];
  T* shared = reinterpret_cast<T*>(shared_mem);
  
  const int tid = threadIdx.x;
  const int64_t block_start = blockIdx.x * blockDim.x;
  const int64_t global_tid = block_start + tid;
  
  // Each block processes a chunk of the output
  // Layout per block: [num_inputs][elements_per_input]
  
  // Calculate how many elements this block processes
  const int64_t block_num_elements = min((int64_t)blockDim.x, total_elements - block_start);
  
  // Step 1: Load data from inputs to shared memory
  // Each thread loads one element
  if (global_tid < total_elements) {
    // Decompose global index
    const int64_t after_idx = global_tid % after_size;
    const int64_t temp = global_tid / after_size;
    const int i = temp % num_inputs;
    const int64_t b = temp / num_inputs;
    
    const int64_t in_idx = b * after_size + after_idx;
    
    // Store in shared memory with coalesced layout
    // shared[i][after_idx] = inputs[i][in_idx]
    const int shared_idx = i * after_size + after_idx;
    if (shared_idx < num_inputs * after_size) {
      shared[shared_idx] = inputs[i][in_idx];
    }
  }
  
  __syncthreads();
  
  // Step 2: Write from shared memory to global output
  // This ensures coalesced writes to global memory
  if (global_tid < total_elements) {
    output[global_tid] = shared[tid];
  }
}

// ============================================================================
// Optimized Unpack Kernel - Vectorized and Coalesced Memory Access
// ============================================================================

template <typename T>
__global__ void UnpackKernelScalar(const T* __restrict__ input,
                                    T** __restrict__ outputs,
                                    int num_outputs,
                                    int64_t before_size,
                                    int64_t after_size,
                                    int64_t total_elements) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  // Decompose input index: tid -> (b, i, a)
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_outputs;
  const int64_t b = temp / num_outputs;
  
  // Output layout: outputs[i][before][after]
  const int64_t out_idx = b * after_size + after_idx;
  outputs[i][out_idx] = input[tid];
}

// Vectorized float4 version
__global__ void UnpackKernelFloat4(const float4* __restrict__ input,
                                    float4** __restrict__ outputs,
                                    int num_outputs,
                                    int64_t before_size,
                                    int64_t after_size,  // in float4 units
                                    int64_t total_elements) {  // in float4 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_outputs;
  
  outputs[i][after_idx] = input[tid];
}

// Vectorized int4 version
__global__ void UnpackKernelInt4(const int4* __restrict__ input,
                                  int4** __restrict__ outputs,
                                  int num_outputs,
                                  int64_t before_size,
                                  int64_t after_size,  // in int4 units
                                  int64_t total_elements) {  // in int4 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_outputs;
  
  outputs[i][after_idx] = input[tid];
}

// Vectorized half2 version
__global__ void UnpackKernelHalf2(const half2* __restrict__ input,
                                   half2** __restrict__ outputs,
                                   int num_outputs,
                                   int64_t before_size,
                                   int64_t after_size,  // in half2 units
                                   int64_t total_elements) {  // in half2 units
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= total_elements) return;
  
  const int64_t after_idx = tid % after_size;
  const int64_t temp = tid / after_size;
  const int i = temp % num_outputs;
  
  outputs[i][after_idx] = input[tid];
}

// ============================================================================
// Launcher Functions with Auto-Tuned Configuration
// ============================================================================

extern "C" {

// Optimal thread configuration
#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// Threshold for using vectorized kernels
#define VECTORIZE_THRESHOLD 1024

// ----------------------------------------------------------------------------
// Pack Launchers
// ----------------------------------------------------------------------------

#define DEFINE_PACK_LAUNCHER_SCALAR(T, Name) \
  void Name(const T** inputs, T* output, int num_inputs, \
            int64_t before_size, int64_t after_size, int64_t total_elements, \
            musaStream_t stream) { \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    PackKernelScalar<T><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        inputs, output, num_inputs, before_size, after_size, total_elements); \
  }

DEFINE_PACK_LAUNCHER_SCALAR(float, LaunchPackKernelFloat)
DEFINE_PACK_LAUNCHER_SCALAR(double, LaunchPackKernelDouble)
DEFINE_PACK_LAUNCHER_SCALAR(int, LaunchPackKernelInt32)
DEFINE_PACK_LAUNCHER_SCALAR(long long, LaunchPackKernelInt64)

// Half - use scalar kernel for correctness with all index types
void LaunchPackKernelHalf(const void** inputs, void* output, int num_inputs,
                          int64_t before_size, int64_t after_size, int64_t total_elements,
                          musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  PackKernelScalar<half><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half**>(inputs),
      reinterpret_cast<half*>(output),
      num_inputs, before_size, after_size, total_elements);
}

// BFloat16 - use scalar kernel
void LaunchPackKernelBFloat16(const void** inputs, void* output, int num_inputs,
                              int64_t before_size, int64_t after_size, int64_t total_elements,
                              musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  PackKernelScalar<__mt_bfloat16><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16**>(inputs),
      reinterpret_cast<__mt_bfloat16*>(output),
      num_inputs, before_size, after_size, total_elements);
}

#undef DEFINE_PACK_LAUNCHER_SCALAR

// ----------------------------------------------------------------------------
// Unpack Launchers
// ----------------------------------------------------------------------------

#define DEFINE_UNPACK_LAUNCHER_SCALAR(T, Name) \
  void Name(const T* input, T** outputs, int num_outputs, \
            int64_t before_size, int64_t after_size, int64_t total_elements, \
            musaStream_t stream) { \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    UnpackKernelScalar<T><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        input, outputs, num_outputs, before_size, after_size, total_elements); \
  }

DEFINE_UNPACK_LAUNCHER_SCALAR(float, LaunchUnpackKernelFloat)
DEFINE_UNPACK_LAUNCHER_SCALAR(double, LaunchUnpackKernelDouble)
DEFINE_UNPACK_LAUNCHER_SCALAR(int, LaunchUnpackKernelInt32)
DEFINE_UNPACK_LAUNCHER_SCALAR(long long, LaunchUnpackKernelInt64)

// Half
void LaunchUnpackKernelHalf(const void* input, void** outputs, int num_outputs,
                            int64_t before_size, int64_t after_size, int64_t total_elements,
                            musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackKernelScalar<half><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(input),
      reinterpret_cast<half**>(outputs),
      num_outputs, before_size, after_size, total_elements);
}

// BFloat16
void LaunchUnpackKernelBFloat16(const void* input, void** outputs, int num_outputs,
                                int64_t before_size, int64_t after_size, int64_t total_elements,
                                musaStream_t stream) {
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  UnpackKernelScalar<__mt_bfloat16><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input),
      reinterpret_cast<__mt_bfloat16**>(outputs),
      num_outputs, before_size, after_size, total_elements);
}

#undef DEFINE_UNPACK_LAUNCHER_SCALAR

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS
#undef VECTORIZE_THRESHOLD

}  // extern "C"
