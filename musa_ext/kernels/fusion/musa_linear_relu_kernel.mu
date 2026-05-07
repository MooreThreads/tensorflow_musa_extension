#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
  *reinterpret_cast<__half*>(p) = __float2half(v);
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&v);
  *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>((*f_ptr) >> 16);
}

template <typename T>
__global__ void BiasAddReluKernel(const T* __restrict__ x,
                                  const T* __restrict__ bias,
                                  T* __restrict__ output,
                                  int64_t n_elements, int64_t n_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < n_elements; i += stride) {
    const float val = LoadFloat(x + i) + LoadFloat(bias + (i % n_cols));
    StoreFloat(output + i, val > 0.0f ? val : 0.0f);
  }
}

template <>
__global__ void BiasAddReluKernel<double>(const double* __restrict__ x,
                                          const double* __restrict__ bias,
                                          double* __restrict__ output,
                                          int64_t n_elements,
                                          int64_t n_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < n_elements; i += stride) {
    const double val = x[i] + bias[i % n_cols];
    output[i] = val > 0.0 ? val : 0.0;
  }
}

template <typename T>
void LaunchBiasAddReluKernel(const T* x, const T* bias, T* output,
                             int64_t n_elements, int64_t n_cols,
                             musaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int64_t grid_size = (n_elements + kBlockSize - 1) / kBlockSize;
  BiasAddReluKernel<T><<<grid_size, kBlockSize, 0, stream>>>(
      x, bias, output, n_elements, n_cols);
}

template void LaunchBiasAddReluKernel<float>(const float*, const float*, float*,
                                             int64_t, int64_t, musaStream_t);
template void LaunchBiasAddReluKernel<Eigen::half>(const Eigen::half*,
                                                   const Eigen::half*,
                                                   Eigen::half*, int64_t,
                                                   int64_t, musaStream_t);
template void LaunchBiasAddReluKernel<bfloat16>(const bfloat16*,
                                                const bfloat16*, bfloat16*,
                                                int64_t, int64_t,
                                                musaStream_t);
template void LaunchBiasAddReluKernel<double>(const double*, const double*,
                                              double*, int64_t, int64_t,
                                              musaStream_t);

}  // namespace musa
}  // namespace tensorflow
