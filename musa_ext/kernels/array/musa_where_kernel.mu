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
__global__ void MusaSelectFlaggedKernel(const T* __restrict__ d_flags,
                                        TIndex* d_selected_indices,
                                        TIndex* d_num_selected_out,
                                        int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items && IsNonZeroValue<T>(d_flags[idx])) {
    unsigned long long pos = atomicAdd(
        reinterpret_cast<unsigned long long*>(d_num_selected_out), 1ULL);
    d_selected_indices[static_cast<int>(pos)] = static_cast<TIndex>(idx);
  }
}

template <typename T, typename TIndex>
musaError_t LaunchMusaSelectFlaggedKernel(const T* input,
                                          TIndex* selected_indices,
                                          TIndex* num_selected_out,
                                          int num_items, musaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((num_items + threads - 1) / threads);
  MusaSelectFlaggedKernel<T, TIndex><<<blocks, threads, 0, stream>>>(
      input, selected_indices, num_selected_out, num_items);
  return musaGetLastError();
}

#define REGISTER_SELECT_FLAGGED(T, TINDEX)                                  \
  template musaError_t LaunchMusaSelectFlaggedKernel<T, TINDEX>(               \
      const T* input, TINDEX* selected_indices, TINDEX* num_selected_out, int, \
      musaStream_t stream)

#define INSTANTIATE_SELECT_FLAGGED_ALL(T)                                       \
  INSTANTIATE_SELECT_FLAGGED(T, int32_t);                                      \
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

// -------- Propagate selected indices into NDIM output kernel --------

template <int NDIM, typename TIndex>
struct StridesPack {
  TIndex v[NDIM];
};

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const StridesPack<NDIM, TIndex> strides,
    const TIndex* __restrict__ selected_indices, int64_t* __restrict__ output) {
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
musaError_t LaunchPropagateWhereIndicesKernel(const TIndex output_rows,
                                              const TIndex* strides_host,
                                              const TIndex* selected_indices,
                                              int64_t* output,
                                              musaStream_t stream) {
  if (output_rows <= static_cast<TIndex>(0)) {
    return musaSuccess;
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
  return musaGetLastError();
}

#define REGISTER_PROPAGATE(NDIM, TINDEX)                             \
  template musaError_t LaunchPropagateWhereIndicesKernel<NDIM, TINDEX>( \
      const TINDEX output_rows, const TINDEX* strides_host,             \
      const TINDEX* selected_indices, int64_t* output, musaStream_t stream)

REGISTER_PROPAGATE(1, int32);
REGISTER_PROPAGATE(2, int32);
REGISTER_PROPAGATE(3, int32);
REGISTER_PROPAGATE(4, int32);
REGISTER_PROPAGATE(5, int32);
REGISTER_PROPAGATE(6, int32);
REGISTER_PROPAGATE(7, int32);
REGISTER_PROPAGATE(8, int32);

REGISTER_PROPAGATE(1, int64);
REGISTER_PROPAGATE(2, int64);
REGISTER_PROPAGATE(3, int64);
REGISTER_PROPAGATE(4, int64);
REGISTER_PROPAGATE(5, int64);
REGISTER_PROPAGATE(6, int64);
REGISTER_PROPAGATE(7, int64);
REGISTER_PROPAGATE(8, int64);

#undef INSTANTIATE_PROPAGATE

}  // namespace musa
}  // namespace tensorflow