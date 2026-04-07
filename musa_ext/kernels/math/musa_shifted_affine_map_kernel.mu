#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

#include "musa_shifted_affine_map_kernel.h"

using bfloat16 = tensorflow::bfloat16;

namespace tensorflow {
namespace musa {

template <typename T>
struct ShiftedAffineMapAccumType {
  using type = float;
};

template <>
struct ShiftedAffineMapAccumType<double> {
  using type = double;
};

__device__ __forceinline__ float LoadValue(const float* p) { return *p; }

__device__ __forceinline__ double LoadValue(const double* p) { return *p; }

__device__ __forceinline__ float LoadValue(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ float LoadValue(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreValue(float* p, float v) { *p = v; }

__device__ __forceinline__ void StoreValue(double* p, double v) { *p = v; }

__device__ __forceinline__ void StoreValue(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ void StoreValue(bfloat16* p, float v) {
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
  uint16_t b_val = *f_ptr >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__global__ void ShiftedAffineMapKernel(
    const T* data_left, ShiftedAffineMapStrides data_left_st,
    const T* mask, ShiftedAffineMapStrides mask_st,
    const T* sliced_var_right, ShiftedAffineMapStrides sliced_var_right_st,
    T* output, ShiftedAffineMapShape shape, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  int remaining = idx;
  int data_left_offset = 0;
  int mask_offset = 0;
  int sliced_var_right_offset = 0;

  for (int dim = shape.rank - 1; dim >= 0; --dim) {
    const int coord = remaining % shape.dims[dim];
    remaining /= shape.dims[dim];

    data_left_offset += coord * data_left_st.values[dim];
    mask_offset += coord * mask_st.values[dim];
    sliced_var_right_offset += coord * sliced_var_right_st.values[dim];
  }

  using AccT = typename ShiftedAffineMapAccumType<T>::type;
  const AccT data_val = static_cast<AccT>(LoadValue(data_left + data_left_offset));
  const AccT mask_val = static_cast<AccT>(LoadValue(mask + mask_offset));
  const AccT right_val =
      static_cast<AccT>(LoadValue(sliced_var_right + sliced_var_right_offset));

  const AccT result = mask_val * data_val + right_val;
  StoreValue(output + idx, result);
}

template <typename T>
void LaunchShiftedAffineMapKernel(
    const T* data_left, ShiftedAffineMapStrides data_left_st,
    const T* mask, ShiftedAffineMapStrides mask_st,
    const T* sliced_var_right, ShiftedAffineMapStrides sliced_var_right_st,
    T* output, ShiftedAffineMapShape shape, int total_elements,
    musaStream_t stream) {
  if (total_elements == 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (total_elements + block_size - 1) / block_size;
  ShiftedAffineMapKernel<T><<<grid_size, block_size, 0, stream>>>(
      data_left, data_left_st, mask, mask_st, 
      sliced_var_right, sliced_var_right_st, output, shape,
      total_elements);
}

template void LaunchShiftedAffineMapKernel<float>(
    const float*, ShiftedAffineMapStrides, const float*, ShiftedAffineMapStrides,
    const float*, ShiftedAffineMapStrides, float*, ShiftedAffineMapShape, int,
    musaStream_t);
template void LaunchShiftedAffineMapKernel<double>(
    const double*, ShiftedAffineMapStrides, const double*, ShiftedAffineMapStrides,
    const double*, ShiftedAffineMapStrides, double*, ShiftedAffineMapShape,
    int, musaStream_t);
template void LaunchShiftedAffineMapKernel<Eigen::half>(
    const Eigen::half*, ShiftedAffineMapStrides, const Eigen::half*, ShiftedAffineMapStrides,
    const Eigen::half*, ShiftedAffineMapStrides, Eigen::half*,
    ShiftedAffineMapShape, int, musaStream_t);
template void LaunchShiftedAffineMapKernel<bfloat16>(
    const bfloat16*, ShiftedAffineMapStrides, const bfloat16*, ShiftedAffineMapStrides,
    const bfloat16*, ShiftedAffineMapStrides, bfloat16*,
    ShiftedAffineMapShape, int, musaStream_t);

template <typename T>
__global__ void ShiftedAffineMapContiguousKernel(
    const T* data_left, const T* mask, const T* sliced_var_right,
    T* output, int64_t total_elements) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  using AccT = typename ShiftedAffineMapAccumType<T>::type;
  const AccT data_val = static_cast<AccT>(LoadValue(data_left + idx));
  const AccT mask_val = static_cast<AccT>(LoadValue(mask + idx));
  const AccT right_val = static_cast<AccT>(LoadValue(sliced_var_right + idx));

  const AccT result = mask_val * data_val + right_val;
  StoreValue(output + idx, result);
}

template <typename T>
void LaunchShiftedAffineMapContiguous(
    const T* data_left, const T* mask, const T* sliced_var_right,
    T* output, int64_t total_elements,
    musaStream_t stream) {
  if (total_elements <= 0) {
    return;
  }

  const int block_size = 256;
  const int64_t grid_size = (total_elements + block_size - 1) / block_size;
  ShiftedAffineMapContiguousKernel<T><<<grid_size, block_size, 0, stream>>>(
      data_left, mask, sliced_var_right, output, total_elements);
}

template void LaunchShiftedAffineMapContiguous<float>(
    const float*, const float*, const float*, float*, int64_t, musaStream_t);
template void LaunchShiftedAffineMapContiguous<double>(
    const double*, const double*, const double*, double*, int64_t, musaStream_t);
template void LaunchShiftedAffineMapContiguous<Eigen::half>(
    const Eigen::half*, const Eigen::half*, const Eigen::half*, Eigen::half*, int64_t, musaStream_t);
template void LaunchShiftedAffineMapContiguous<bfloat16>(
    const bfloat16*, const bfloat16*, const bfloat16*, bfloat16*, int64_t, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
