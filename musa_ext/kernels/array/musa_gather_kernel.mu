// High-Performance MUSA Gather Kernels
// Optimized for memory bandwidth and coalesced access patterns
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// Gather V2 (axis-based gather) - Optimized Kernel
// ============================================================================

// Optimized gather kernel - scalar version for all types
// params: input tensor data
// indices: index tensor data  
// output: output tensor data
// batch_size: number of batches (outer dimensions before axis)
// axis_size: size of the gather axis dimension (for bounds check)
// inner_size: size of inner dimensions (after axis), in elements
// indices_size: number of indices to gather
// params_stride: stride for params along batch dimension, in elements
// limit: bounds check limit (axis_size)

template <typename T, typename IndexT>
__global__ void GatherV2Kernel(
    const T* __restrict__ params,
    const IndexT* __restrict__ indices,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t axis_size,
    int64_t inner_size,
    int64_t indices_size,
    int64_t params_stride,
    IndexT limit) {
  
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = batch_size * indices_size * inner_size;
  
  if (tid >= total_elements) return;
  
  // Decompose tid into batch, indices_idx, inner_idx coordinates
  // Layout: output[batch][indices_idx][inner]
  const int64_t inner_idx = tid % inner_size;
  const int64_t temp = tid / inner_size;
  const int64_t indices_idx = temp % indices_size;
  const int64_t batch_idx = temp / indices_size;
  
  // Get index from indices tensor
  IndexT idx = indices[indices_idx];
  
  // Clamp index to valid range (GPU-side bounds checking)
  // Note: negative indices are clamped to 0
  if (idx < 0) idx = 0;
  if (idx >= limit) idx = limit - 1;
  
  // Calculate source and destination addresses
  // Source: params[batch][idx][inner]
  const int64_t src_offset = batch_idx * params_stride + idx * inner_size + inner_idx;
  const int64_t dst_offset = tid;
  
  output[dst_offset] = params[src_offset];
}

// ============================================================================
// Gather ND - Optimized Kernel
// ============================================================================

template <typename T, typename IndexT>
__global__ void GatherNDKernel(
    const T* __restrict__ params,
    const IndexT* __restrict__ indices,
    T* __restrict__ output,
    int index_depth,
    int64_t indices_nd_size,
    int64_t slice_size,
    const int64_t* __restrict__ params_strides) {
  
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = indices_nd_size * slice_size;
  
  if (tid >= total_elements) return;
  
  // Decompose: tid -> (indices_idx, slice_idx)
  const int64_t slice_idx = tid % slice_size;
  const int64_t indices_idx = tid / slice_size;
  
  // Compute params offset from indices
  int64_t params_offset = slice_idx;
  
  #pragma unroll
  for (int d = 0; d < index_depth; ++d) {
    IndexT idx = indices[indices_idx * index_depth + d];
    params_offset += (int64_t)idx * params_strides[d];
  }
  
  output[tid] = params[params_offset];
}

// ============================================================================
// Resource Gather - Uses Gather V2 internally
// ============================================================================

template <typename T, typename IndexT>
__global__ void ResourceGatherKernel(
    const T* __restrict__ params,
    const IndexT* __restrict__ indices,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t inner_size,
    int64_t indices_size,
    int64_t params_stride,
    IndexT limit) {
  
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = batch_size * indices_size * inner_size;
  
  if (tid >= total_elements) return;
  
  const int64_t inner_idx = tid % inner_size;
  const int64_t temp = tid / inner_size;
  const int64_t indices_idx = temp % indices_size;
  const int64_t batch_idx = temp / indices_size;
  
  IndexT idx = indices[indices_idx];
  
  // Fast bounds check - clamp to valid range
  if (idx < 0) idx = 0;
  if (idx >= limit) idx = limit - 1;
  
  const int64_t src_offset = batch_idx * params_stride + idx * inner_size + inner_idx;
  output[tid] = params[src_offset];
}

// ============================================================================
// Launcher Functions
// ============================================================================

extern "C" {

// Optimal thread configuration
#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// ----------------------------------------------------------------------------
// Gather V2 Launchers - Scalar versions for all types
// ----------------------------------------------------------------------------

#define DEFINE_GATHER_V2_LAUNCHER(T, IndexT, Name) \
  void Name(const T* params, const IndexT* indices, T* output, \
            int64_t batch_size, int64_t axis_size, int64_t inner_size, \
            int64_t indices_size, int64_t params_stride, IndexT limit, \
            musaStream_t stream) { \
    const int64_t total_elements = batch_size * indices_size * inner_size; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    GatherV2Kernel<T, IndexT><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        params, indices, output, batch_size, axis_size, inner_size, \
        indices_size, params_stride, limit); \
  }

DEFINE_GATHER_V2_LAUNCHER(float, int, LaunchGatherV2FloatInt32)
DEFINE_GATHER_V2_LAUNCHER(float, long long, LaunchGatherV2FloatInt64)
DEFINE_GATHER_V2_LAUNCHER(double, int, LaunchGatherV2DoubleInt32)
DEFINE_GATHER_V2_LAUNCHER(double, long long, LaunchGatherV2DoubleInt64)
DEFINE_GATHER_V2_LAUNCHER(int, int, LaunchGatherV2Int32Int32)
DEFINE_GATHER_V2_LAUNCHER(int, long long, LaunchGatherV2Int32Int64)
DEFINE_GATHER_V2_LAUNCHER(long long, int, LaunchGatherV2Int64Int32)
DEFINE_GATHER_V2_LAUNCHER(long long, long long, LaunchGatherV2Int64Int64)
DEFINE_GATHER_V2_LAUNCHER(bool, int, LaunchGatherV2BoolInt32)
DEFINE_GATHER_V2_LAUNCHER(bool, long long, LaunchGatherV2BoolInt64)

// Half (Eigen::half maps to half) - use scalar kernel for correctness
void LaunchGatherV2HalfInt32(const void* params, const int* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int limit,
                             musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherV2Kernel<half, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), batch_size, axis_size, inner_size,
      indices_size, params_stride, limit);
}

void LaunchGatherV2HalfInt64(const void* params, const long long* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, long long limit,
                             musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherV2Kernel<half, long long><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), batch_size, axis_size, inner_size,
      indices_size, params_stride, limit);
}

// BFloat16 - use scalar kernel
void LaunchGatherV2BFloat16Int32(const void* params, const int* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, int limit,
                                 musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherV2Kernel<__mt_bfloat16, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(params), indices,
      reinterpret_cast<__mt_bfloat16*>(output), batch_size, axis_size, inner_size,
      indices_size, params_stride, limit);
}

void LaunchGatherV2BFloat16Int64(const void* params, const long long* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, long long limit,
                                 musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherV2Kernel<__mt_bfloat16, long long><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(params), indices,
      reinterpret_cast<__mt_bfloat16*>(output), batch_size, axis_size, inner_size,
      indices_size, params_stride, limit);
}

#undef DEFINE_GATHER_V2_LAUNCHER

// ----------------------------------------------------------------------------
// Gather ND Launchers
// ----------------------------------------------------------------------------

#define DEFINE_GATHER_ND_LAUNCHER(T, IndexT, Name) \
  void Name(const T* params, const IndexT* indices, T* output, \
            int index_depth, int64_t indices_nd_size, int64_t slice_size, \
            const int64_t* params_strides, musaStream_t stream) { \
    const int64_t total_elements = indices_nd_size * slice_size; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    GatherNDKernel<T, IndexT><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        params, indices, output, index_depth, indices_nd_size, slice_size, params_strides); \
  }

DEFINE_GATHER_ND_LAUNCHER(float, int, LaunchGatherNDFloatInt32)
DEFINE_GATHER_ND_LAUNCHER(float, long long, LaunchGatherNDFloatInt64)
DEFINE_GATHER_ND_LAUNCHER(double, int, LaunchGatherNDDoubleInt32)
DEFINE_GATHER_ND_LAUNCHER(double, long long, LaunchGatherNDDoubleInt64)
DEFINE_GATHER_ND_LAUNCHER(int, int, LaunchGatherNDInt32Int32)
DEFINE_GATHER_ND_LAUNCHER(int, long long, LaunchGatherNDInt32Int64)
DEFINE_GATHER_ND_LAUNCHER(long long, int, LaunchGatherNDInt64Int32)
DEFINE_GATHER_ND_LAUNCHER(long long, long long, LaunchGatherNDInt64Int64)
DEFINE_GATHER_ND_LAUNCHER(bool, int, LaunchGatherNDBoolInt32)
DEFINE_GATHER_ND_LAUNCHER(bool, long long, LaunchGatherNDBoolInt64)

// Half
void LaunchGatherNDHalfInt32(const void* params, const int* indices, void* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream) {
  const int64_t total_elements = indices_nd_size * slice_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherNDKernel<half, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), index_depth, indices_nd_size, slice_size, params_strides);
}

void LaunchGatherNDHalfInt64(const void* params, const long long* indices, void* output,
                             int index_depth, int64_t indices_nd_size, int64_t slice_size,
                             const int64_t* params_strides, musaStream_t stream) {
  const int64_t total_elements = indices_nd_size * slice_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherNDKernel<half, long long><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), index_depth, indices_nd_size, slice_size, params_strides);
}

// BFloat16
void LaunchGatherNDBFloat16Int32(const void* params, const int* indices, void* output,
                                 int index_depth, int64_t indices_nd_size, int64_t slice_size,
                                 const int64_t* params_strides, musaStream_t stream) {
  const int64_t total_elements = indices_nd_size * slice_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherNDKernel<__mt_bfloat16, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(params), indices,
      reinterpret_cast<__mt_bfloat16*>(output), index_depth, indices_nd_size, slice_size, params_strides);
}

void LaunchGatherNDBFloat16Int64(const void* params, const long long* indices, void* output,
                                 int index_depth, int64_t indices_nd_size, int64_t slice_size,
                                 const int64_t* params_strides, musaStream_t stream) {
  const int64_t total_elements = indices_nd_size * slice_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  GatherNDKernel<__mt_bfloat16, long long><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(params), indices,
      reinterpret_cast<__mt_bfloat16*>(output), index_depth, indices_nd_size, slice_size, params_strides);
}

#undef DEFINE_GATHER_ND_LAUNCHER

// ----------------------------------------------------------------------------
// Resource Gather Launchers
// ----------------------------------------------------------------------------

#define DEFINE_RESOURCE_GATHER_LAUNCHER(T, IndexT, Name) \
  void Name(const T* params, const IndexT* indices, T* output, \
            int64_t batch_size, int64_t inner_size, int64_t indices_size, \
            int64_t params_stride, IndexT limit, musaStream_t stream) { \
    const int64_t total_elements = batch_size * indices_size * inner_size; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    ResourceGatherKernel<T, IndexT><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        params, indices, output, batch_size, inner_size, indices_size, \
        params_stride, limit); \
  }

DEFINE_RESOURCE_GATHER_LAUNCHER(float, int, LaunchResourceGatherFloatInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(float, long long, LaunchResourceGatherFloatInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, int, LaunchResourceGatherDoubleInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, long long, LaunchResourceGatherDoubleInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(int, int, LaunchResourceGatherInt32Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(int, long long, LaunchResourceGatherInt32Int64)
DEFINE_RESOURCE_GATHER_LAUNCHER(long long, int, LaunchResourceGatherInt64Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(long long, long long, LaunchResourceGatherInt64Int64)

// Half
void LaunchResourceGatherHalfInt32(const void* params, const int* indices, void* output,
                                   int64_t batch_size, int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, int limit, musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  ResourceGatherKernel<half, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), batch_size, inner_size, indices_size,
      params_stride, limit);
}

void LaunchResourceGatherHalfInt64(const void* params, const long long* indices, void* output,
                                   int64_t batch_size, int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, long long limit, musaStream_t stream) {
  const int64_t total_elements = batch_size * indices_size * inner_size;
  if (total_elements == 0) return;
  const int blocks = OPTIMAL_BLOCKS(total_elements);
  ResourceGatherKernel<half, long long><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(params), indices,
      reinterpret_cast<half*>(output), batch_size, inner_size, indices_size,
      params_stride, limit);
}

#undef DEFINE_RESOURCE_GATHER_LAUNCHER

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"
