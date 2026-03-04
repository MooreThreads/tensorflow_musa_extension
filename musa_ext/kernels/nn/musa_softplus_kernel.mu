#include <math.h>
#include <stdint.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"

#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

// ===================== Load / Store Helpers =====================

// float
__device__ __forceinline__ float LoadAsFloat(const float* p) { return *p; }

__device__ __forceinline__ void StoreFromFloat(float v, float* p) { *p = v; }

// half (Eigen::half <-> __half)
__device__ __forceinline__ float LoadAsFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFromFloat(float v, Eigen::half* p) {
  __half hv = __float2half(v);
  *reinterpret_cast<__half*>(p) = hv;
}

// bfloat16
__device__ __forceinline__ float LoadAsFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFromFloat(float v, bfloat16* p) {
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&v);
  uint16_t* dst = reinterpret_cast<uint16_t*>(p);
  *dst = static_cast<uint16_t>((*src) >> 16);
}

// ===================== Stable Softplus =====================

// stable softplus for float:
// softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
__device__ __forceinline__ float SoftplusStable(float x) {
  float ax = fabsf(x);
  float e = expf(-ax);              // <= 1, no overflow
  float m = fmaxf(x, 0.0f);
  return m + logf(1.0f + e);
}


// ===================== Kernels =====================

// Generic kernel for float-like types using float accumulation/store
template <typename T>
__global__ void SoftplusKernel(const T* input, T* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = LoadAsFloat(&input[idx]);
    float y = SoftplusStable(x);
    StoreFromFloat(y, &output[idx]);
  }
}


// ===================== Launch =====================

template <typename T>
void LaunchSoftplus(const T* input, T* output, int n, musaStream_t stream) {
  if (n <= 0) return;

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  SoftplusKernel<T><<<blocks, threads, 0, stream>>>(input, output, n);

  musaError_t err = musaGetLastError();
  (void)err;
}

template void LaunchSoftplus<float>(const float*, float*, int, musaStream_t);
template void LaunchSoftplus<Eigen::half>(const Eigen::half*, Eigen::half*, int,
                                          musaStream_t);
template void LaunchSoftplus<bfloat16>(const bfloat16*, bfloat16*, int,
                                       musaStream_t);


}  // namespace musa
}  // namespace tensorflow