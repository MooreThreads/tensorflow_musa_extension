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
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  uint16_t* b_ptr = (uint16_t*)p;
  uint32_t* f_ptr = (uint32_t*)&res;
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  uint32_t* f_ptr = (uint32_t*)&v;
  uint16_t b_val = (*f_ptr) >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__global__ void BiasAddReluKernel(const T* x, const T* bias, T* output,
                                  int n_elements, int n_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    int col = idx % n_cols;
    float val = LoadFloat(&x[idx]) + LoadFloat(&bias[col]);
    StoreFloat(&output[idx], val > 0.0f ? val : 0.0f);
  }
}

template <>
__global__ void BiasAddReluKernel<double>(const double* x, const double* bias,
                                          double* output, int n_elements,
                                          int n_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    output[idx] = x[idx] + bias[idx % n_cols];
  }
}

template <typename T>
void LaunchBiasAddReluKernel(const T* x, const T* bias, T* output,
                             int n_elements, int n_cols, musaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n_elements + block_size - 1) / block_size;
  BiasAddReluKernel<T><<<num_blocks, block_size, 0, stream>>>(
      x, bias, output, n_elements, n_cols);
}

template void LaunchBiasAddReluKernel<float>(const float*, const float*, float*,
                                             int, int, musaStream_t);
template void LaunchBiasAddReluKernel<Eigen::half>(const Eigen::half*,
                                                   const Eigen::half*,
                                                   Eigen::half*, int, int,
                                                   musaStream_t);
template void LaunchBiasAddReluKernel<bfloat16>(const bfloat16*,
                                                const bfloat16*, bfloat16*, int,
                                                int, musaStream_t);
template void LaunchBiasAddReluKernel<double>(const double*, const double*,
                                              double*, int, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
