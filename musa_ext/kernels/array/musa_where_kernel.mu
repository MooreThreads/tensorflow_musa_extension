#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace musa {

// -------- Select indices of true values kernel --------

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

template <typename T>
__device__ __forceinline__ bool IsNonZeroValue(const T& v) {
  return v != static_cast<T>(0);
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<float>(const float& v) {
  return v != 0.0f;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<double>(const double& v) {
  return v != 0.0;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<Eigen::half>(
    const Eigen::half& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<bfloat16>(const bfloat16& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <typename T, typename TIndex>
__global__ void MusaMarkFlaggedKernel(const T* __restrict__ d_flags,
                                      TIndex* d_marks, int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    d_marks[idx] = IsNonZeroValue<T>(d_flags[idx]) ? 1 : 0;
  }
}

template <typename TIndex>
__global__ void MusaBlockScanKernel(const TIndex* input, TIndex* output,
                                    TIndex* block_sums, int n) {
  extern __shared__ char shared_mem[];
  TIndex* temp = reinterpret_cast<TIndex*>(shared_mem);
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load into shared memory
  temp[tid] = (idx < n) ? input[idx] : 0;
  __syncthreads();

  // Simple inclusive scan within block
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    TIndex val = 0;
    if (tid >= stride) val = temp[tid - stride];
    __syncthreads();
    temp[tid] += val;
    __syncthreads();
  }

  if (idx < n) output[idx] = temp[tid];
  if (tid == blockDim.x - 1 && block_sums) {
    block_sums[blockIdx.x] = temp[tid];
  }
}

template <typename TIndex>
__global__ void MusaAddBlockOffsetsKernel(TIndex* output,
                                          const TIndex* block_offsets, int n) {
  int bid = blockIdx.x;
  if (bid == 0) return;
  int idx = bid * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] += block_offsets[bid - 1];
  }
}

template <typename TIndex>
__global__ void MusaScatterIndicesKernel(const TIndex* __restrict__ d_marks,
                                          const TIndex* __restrict__ d_scanned,
                                          TIndex* d_selected_indices,
                                          int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items && d_marks[idx] == 1) {
    // d_scanned is inclusive sum, so (sum - 1) is the zero-based index for the
    // current item
    TIndex pos = d_scanned[idx] - 1;
    if (d_selected_indices) {
      d_selected_indices[static_cast<TIndex>(pos)] = static_cast<TIndex>(idx);
    }
  }
}

template <typename T, typename TIndex>
void LaunchMusaSelectFlaggedKernel(const T* input, TIndex* selected_indices,
                                   TIndex* num_selected_out, int num_items,
                                   musaStream_t stream) {
  if (num_items <= 0) return;

  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;

  // 1. Mark elements (0 or 1)
  TIndex* d_marks = nullptr;
  musaMalloc(reinterpret_cast<void**>(&d_marks),
                  num_items * sizeof(TIndex));
  MusaMarkFlaggedKernel<T, TIndex>
      <<<blocks, threads, 0, stream>>>(input, d_marks, num_items);

  // 2. Multi-pass Global Scan (Inclusive)
  TIndex* d_scanned = nullptr;
  musaMalloc(reinterpret_cast<void**>(&d_scanned),
                  num_items * sizeof(TIndex));

  if (blocks == 1) {
    MusaBlockScanKernel<TIndex><<<1, threads, threads * sizeof(TIndex), stream>>>(
        d_marks, d_scanned, nullptr, num_items);
  } else {
    TIndex* d_block_sums = nullptr;
    musaMalloc(reinterpret_cast<void**>(&d_block_sums),
                    blocks * sizeof(TIndex));

    // Pass 1: Local block scan
    MusaBlockScanKernel<TIndex>
        <<<blocks, threads, threads * sizeof(TIndex), stream>>>(
            d_marks, d_scanned, d_block_sums, num_items);

    // Pass 2: Scan the block sums (assume blocks < threads for simplicity in
    // this example)
    TIndex* d_block_offsets = nullptr;
    musaMalloc(reinterpret_cast<void**>(&d_block_offsets),
                    blocks * sizeof(TIndex));
    MusaBlockScanKernel<TIndex><<<1, threads, threads * sizeof(TIndex), stream>>>(
        d_block_sums, d_block_offsets, nullptr, blocks);

    // Pass 3: Add offsets back
    MusaAddBlockOffsetsKernel<TIndex>
        <<<blocks, threads, 0, stream>>>(d_scanned, d_block_offsets, num_items);

    musaFree(d_block_sums);
    musaFree(d_block_offsets);
  }

  // 3. Scatter indices keeping original order
  MusaScatterIndicesKernel<<<blocks, threads, 0, stream>>>(
      d_marks, d_scanned, selected_indices, num_items);

  // 4. Update total count
  if (num_selected_out) {
    musaMemcpyAsync(num_selected_out, d_scanned + num_items - 1, sizeof(TIndex),
                    musaMemcpyDeviceToDevice, stream);
  }

  musaFree(d_marks);
  musaFree(d_scanned);
}

#define INSTANTIATE_SELECT_FLAGGED(T, TINDEX)                                  \
  template void LaunchMusaSelectFlaggedKernel<T, TINDEX>(                      \
      const T* input, TINDEX* selected_indices, TINDEX* num_selected_out, int, \
      musaStream_t stream)

#define INSTANTIATE_SELECT_FLAGGED_ALL(T) \
  INSTANTIATE_SELECT_FLAGGED(T, int32_t); \
  INSTANTIATE_SELECT_FLAGGED(T, int64_t); \
  INSTANTIATE_SELECT_FLAGGED(T, long long)

INSTANTIATE_SELECT_FLAGGED_ALL(bool);
INSTANTIATE_SELECT_FLAGGED_ALL(float);
INSTANTIATE_SELECT_FLAGGED_ALL(double);
INSTANTIATE_SELECT_FLAGGED_ALL(int8);
INSTANTIATE_SELECT_FLAGGED_ALL(uint8);
INSTANTIATE_SELECT_FLAGGED_ALL(int16);
INSTANTIATE_SELECT_FLAGGED_ALL(uint16);
INSTANTIATE_SELECT_FLAGGED_ALL(int32);
INSTANTIATE_SELECT_FLAGGED_ALL(int64);
INSTANTIATE_SELECT_FLAGGED_ALL(bfloat16);

// -------- Propagate selected indices into NDIM output kernel --------

template <int NDIM, typename TIndex>
struct StridesPack {
  TIndex v[NDIM];
};

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const StridesPack<NDIM, TIndex> strides,
    const TIndex* __restrict__ selected_indices, TIndex* __restrict__ output) {
  const TIndex i = static_cast<TIndex>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < output_rows) {
    TIndex index_value = selected_indices[i];
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      const TIndex stride = strides.v[c];
      *(output + NDIM * i + c) = index_value / stride;
      index_value %= stride;
    }
  }
}

template <int NDIM, typename TIndex>
void LaunchPropagateWhereIndicesKernel(const TIndex output_rows,
                                       const TIndex* strides_host,
                                       const TIndex* selected_indices,
                                       TIndex* output, musaStream_t stream) {
  if (output_rows <= static_cast<TIndex>(0)) {
    return;
  }

  StridesPack<NDIM, TIndex> pack;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    pack.v[i] = strides_host[i];
  }

  const int block_size = 256;
  const int grid_size =
      static_cast<int>((output_rows + block_size - 1) / block_size);
  PropagateWhereIndicesKernel<NDIM, TIndex>
      <<<grid_size, block_size, 0, stream>>>(output_rows, pack,
                                             selected_indices, output);
}

#define INSTANTIATE_PROPAGATE(NDIM, TINDEX)                      \
  template void LaunchPropagateWhereIndicesKernel<NDIM, TINDEX>( \
      const TINDEX output_rows, const TINDEX* strides_host,      \
      const TINDEX* selected_indices, TINDEX* output, musaStream_t stream)

INSTANTIATE_PROPAGATE(1, int32);
INSTANTIATE_PROPAGATE(2, int32);
INSTANTIATE_PROPAGATE(3, int32);
INSTANTIATE_PROPAGATE(4, int32);
INSTANTIATE_PROPAGATE(5, int32);
INSTANTIATE_PROPAGATE(6, int32);
INSTANTIATE_PROPAGATE(7, int32);
INSTANTIATE_PROPAGATE(8, int32);

INSTANTIATE_PROPAGATE(1, int64);
INSTANTIATE_PROPAGATE(2, int64);
INSTANTIATE_PROPAGATE(3, int64);
INSTANTIATE_PROPAGATE(4, int64);
INSTANTIATE_PROPAGATE(5, int64);
INSTANTIATE_PROPAGATE(6, int64);
INSTANTIATE_PROPAGATE(7, int64);
INSTANTIATE_PROPAGATE(8, int64);

#undef INSTANTIATE_PROPAGATE

}  // namespace musa
}  // namespace tensorflow