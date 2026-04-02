#include <math.h>
#include <mudnn.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename T>
__device__ __forceinline__ bool IsNan(T v) {
  return isnan(static_cast<float>(v));
}
template <>
__device__ __forceinline__ bool IsNan<double>(double v) {
  return isnan(v);
}
template <>
__device__ __forceinline__ bool IsNan<int32_t>(int32_t v) {
  return false;
}
template <>
__device__ __forceinline__ bool IsNan<int64_t>(int64_t v) {
  return false;
}

template <>
__device__ __forceinline__ bool IsNan<Eigen::half>(Eigen::half v) {
  return isnan(__half2float(*(const __half*)&v));
}
template <>
__device__ __forceinline__ bool IsNan<Eigen::bfloat16>(Eigen::bfloat16 v) {
  uint16_t b = *(const uint16_t*)&v;
  uint32_t f = static_cast<uint32_t>(b) << 16;
  return isnan(*(const float*)&f);
}

template <typename T>
__device__ __forceinline__ bool LessThan(T a, T b) {
  return a < b;
}
template <>
__device__ __forceinline__ bool LessThan<Eigen::half>(Eigen::half a,
                                                      Eigen::half b) {
  return __half2float(*(const __half*)&a) < __half2float(*(const __half*)&b);
}
template <>
__device__ __forceinline__ bool LessThan<Eigen::bfloat16>(Eigen::bfloat16 a,
                                                          Eigen::bfloat16 b) {
  uint16_t b1 = *(const uint16_t*)&a;
  uint32_t f1 = static_cast<uint32_t>(b1) << 16;
  uint16_t b2 = *(const uint16_t*)&b;
  uint32_t f2 = static_cast<uint32_t>(b2) << 16;
  return (*(const float*)&f1) < (*(const float*)&f2);
}

template <typename T>
__device__ __forceinline__ bool GreaterThan(T a, T b) {
  return a > b;
}
template <>
__device__ __forceinline__ bool GreaterThan<Eigen::half>(Eigen::half a,
                                                         Eigen::half b) {
  return __half2float(*(const __half*)&a) > __half2float(*(const __half*)&b);
}
template <>
__device__ __forceinline__ bool GreaterThan<Eigen::bfloat16>(
    Eigen::bfloat16 a, Eigen::bfloat16 b) {
  uint16_t b1 = *(const uint16_t*)&a;
  uint32_t f1 = static_cast<uint32_t>(b1) << 16;
  uint16_t b2 = *(const uint16_t*)&b;
  uint32_t f2 = static_cast<uint32_t>(b2) << 16;
  return (*(const float*)&f1) > (*(const float*)&f2);
}

template <typename T>
__device__ __forceinline__ void SetZero(T* p) {
  *p = T(0);
}
template <>
__device__ __forceinline__ void SetZero<Eigen::half>(Eigen::half* p) {
  *(uint16_t*)p = 0;
}
template <>
__device__ __forceinline__ void SetZero<Eigen::bfloat16>(Eigen::bfloat16* p) {
  *(uint16_t*)p = 0;
}

template <typename T>
__global__ void SafeClipKernel(const T* x, const T* lo, const T* hi, T* y,
                               int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val_x = x[idx];
    if (IsNan(val_x)) {
      SetZero(&y[idx]);
    } else {
      T val_lo = lo[0];
      T val_hi = hi[0];
      T res = val_x;
      if (LessThan(res, val_lo)) res = val_lo;
      if (GreaterThan(res, val_hi)) res = val_hi;
      y[idx] = res;
    }
  }
}

}  // namespace

template <typename T>
void LaunchSafeClip(const musaStream_t stream, const T* x_ptr, const int n,
                    const T* lo_ptr, const T* hi_ptr, T* y_ptr) {
  if (n == 0) return;

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  SafeClipKernel<T>
      <<<grid_size, block_size, 0, stream>>>(x_ptr, lo_ptr, hi_ptr, y_ptr, n);
}

#define DEFINE_SAFE_CLIP_FUNCTOR(T)                                          \
  template void LaunchSafeClip<T>(const musaStream_t stream, const T* x_ptr, \
                                  const int n, const T* lo_ptr,              \
                                  const T* hi_ptr, T* y_ptr);

DEFINE_SAFE_CLIP_FUNCTOR(float);
DEFINE_SAFE_CLIP_FUNCTOR(double);
DEFINE_SAFE_CLIP_FUNCTOR(int32);
DEFINE_SAFE_CLIP_FUNCTOR(int64);
DEFINE_SAFE_CLIP_FUNCTOR(Eigen::half);
DEFINE_SAFE_CLIP_FUNCTOR(bfloat16);

}  // namespace musa
}  // namespace tensorflow
