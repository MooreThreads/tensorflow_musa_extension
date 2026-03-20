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
__global__ void MusaScatterIndicesKernel(const TIndex* __restrict__ d_marks,
                                         const TIndex* __restrict__ d_scanned,
                                         TIndex* d_selected_indices,
                                         int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items && d_marks[idx] == 1) {
    // d_scanned[idx] is the inclusive sum of marks, i.e., count of 1s in [0, idx].
    TIndex pos = d_scanned[idx] - 1;
    if (d_selected_indices && pos >= 0) {
      d_selected_indices[pos] = static_cast<TIndex>(idx);
    }
  }
}

template <typename T, typename TIndex>
void LaunchMusaSelectFlaggedKernel(const T* input, TIndex* selected_indices,
                                   const TIndex* d_scanned,
                                   const TIndex* d_marks, int num_items,
                                   musaStream_t stream) {
  if (num_items <= 0) return;

  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;

  // Scatter indices keeping original order using the provided prefix sum
  // (d_scanned)
  MusaScatterIndicesKernel<<<blocks, threads, 0, stream>>>(
      d_marks, d_scanned, selected_indices, num_items);
}

// Wrapper to launch Mark kernel separately since muDNN needs to be in .h/.cc
template <typename T, typename TIndex>
void LaunchMusaMarkFlaggedKernel(const T* input, TIndex* d_marks, int num_items,
                                 musaStream_t stream) {
  if (num_items <= 0) return;
  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;
  MusaMarkFlaggedKernel<T, TIndex>
      <<<blocks, threads, 0, stream>>>(input, d_marks, num_items);
}

#define INSTANTIATE_SELECT_FLAGGED(T, TINDEX)                            \
  template void LaunchMusaSelectFlaggedKernel<T, TINDEX>(                \
      const T* input, TINDEX* selected_indices, const TINDEX* d_scanned, \
      const TINDEX* d_marks, int num_items, musaStream_t stream);        \
  template void LaunchMusaMarkFlaggedKernel<T, TINDEX>(                  \
      const T* input, TINDEX* d_marks, int num_items, musaStream_t stream)

#define INSTANTIATE_SELECT_FLAGGED_ALL(T) \
  INSTANTIATE_SELECT_FLAGGED(T, int32_t); \
  INSTANTIATE_SELECT_FLAGGED(T, int64_t)

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
#undef INSTANTIATE_SELECT_FLAGGED
#undef INSTANTIATE_SELECT_FLAGGED_ALL

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