#include <math.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr float kHalf = 0.5f;
constexpr float kRsqrt2 = 0.70710678118f;
constexpr float kApproxCoeff = 0.044715f;
constexpr float kApproxScale = 0.7978845608f;

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

template <typename ScalarT>
__device__ __forceinline__ ScalarT ExactGelu(ScalarT x) {
  return static_cast<ScalarT>(kHalf) * x *
         (static_cast<ScalarT>(1) + erf(x * static_cast<ScalarT>(kRsqrt2)));
}

template <typename ScalarT>
__device__ __forceinline__ ScalarT ApproximateGelu(ScalarT x) {
  const ScalarT x3 = x * x * x;
  const ScalarT inner =
      static_cast<ScalarT>(kApproxScale) *
      (x + static_cast<ScalarT>(kApproxCoeff) * x3);
  return static_cast<ScalarT>(kHalf) * x *
         (static_cast<ScalarT>(1) + tanh(inner));
}

template <typename T>
__global__ void GeluKernel(const T* src, T* dst, int n, bool approximate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float x = LoadFloat(&src[i]);
  const float y = approximate ? ApproximateGelu(x) : ExactGelu(x);
  StoreFloat(&dst[i], y);
}

template <>
__global__ void GeluKernel<double>(const double* src, double* dst, int n,
                                   bool approximate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double x = src[i];
  dst[i] = approximate ? ApproximateGelu(x) : ExactGelu(x);
}

}  // namespace

template <typename T>
void LaunchGelu(const T* src, T* dst, int n, bool approximate,
                musaStream_t stream) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  GeluKernel<T><<<blocks, threads, 0, stream>>>(src, dst, n, approximate);
}

template void LaunchGelu<float>(const float*, float*, int, bool, musaStream_t);
template void LaunchGelu<double>(const double*, double*, int, bool,
                                 musaStream_t);
template void LaunchGelu<Eigen::half>(const Eigen::half*, Eigen::half*, int,
                                      bool, musaStream_t);
template void LaunchGelu<bfloat16>(const bfloat16*, bfloat16*, int, bool,
                                   musaStream_t);

}  // namespace musa
}  // namespace tensorflow
