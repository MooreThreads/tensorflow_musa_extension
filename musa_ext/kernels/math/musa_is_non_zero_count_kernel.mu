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
__device__ __forceinline__ bool IsNonZeroValue<Eigen::half>(const Eigen::half& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<bfloat16>(const bfloat16& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <typename T>
__global__ void IsNonZeroCountKernel(const T* input, int* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && IsNonZeroValue<T>(input[idx])) {
    atomicAdd(output, 1);
  }
}

template <typename T>
void LaunchIsNonZeroCount(const T* input, int* output, int n,
                          musaStream_t stream) {
  if (n <= 0) return;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  IsNonZeroCountKernel<T><<<blocks, threads, 0, stream>>>(input, output, n);
  musaError_t err = musaGetLastError();
  (void)err;
}

template void LaunchIsNonZeroCount<float>(const float*, int*, int,
                                          musaStream_t);
template void LaunchIsNonZeroCount<double>(const double*, int*, int,
                                           musaStream_t);
template void LaunchIsNonZeroCount<Eigen::half>(const Eigen::half*, int*, int,
                                                musaStream_t);
template void LaunchIsNonZeroCount<bfloat16>(const bfloat16*, int*, int,
                                             musaStream_t);
template void LaunchIsNonZeroCount<int8_t>(const int8_t*, int*, int,
                                           musaStream_t);
template void LaunchIsNonZeroCount<uint8_t>(const uint8_t*, int*, int,
                                            musaStream_t);
template void LaunchIsNonZeroCount<int16_t>(const int16_t*, int*, int,
                                            musaStream_t);
template void LaunchIsNonZeroCount<uint16_t>(const uint16_t*, int*, int,
                                             musaStream_t);
template void LaunchIsNonZeroCount<int32_t>(const int32_t*, int*, int,
                                            musaStream_t);
template void LaunchIsNonZeroCount<int64_t>(const int64_t*, int*, int,
                                            musaStream_t);
template void LaunchIsNonZeroCount<bool>(const bool*, int*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow