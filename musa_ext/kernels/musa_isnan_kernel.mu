#include <musa_runtime.h>
#include <musa_fp16.h>
#include <math.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

// --------- 工具：half / bfloat16 转 float ---------
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

// --------- isnan 判定（float/double 直接用 isnan；half/bf16 转 float） ---------
__device__ __forceinline__ bool IsNanValue(float v) { return isnan(v); }
__device__ __forceinline__ bool IsNanValue(double v) { return isnan(v); }

// --------- Kernel：通用模板（float/double）---------
template <typename T>
__global__ void IsNanKernel(const T* input, bool* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = IsNanValue(input[idx]);
  }
}

// --------- 特化：Eigen::half ---------
template <>
__global__ void IsNanKernel<Eigen::half>(const Eigen::half* input, bool* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = LoadFloat(&input[idx]);
    output[idx] = IsNanValue(v);
  }
}

// --------- 特化：bfloat16 ---------
template <>
__global__ void IsNanKernel<bfloat16>(const bfloat16* input, bool* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = LoadFloat(&input[idx]);
    output[idx] = IsNanValue(v);
  }
}

// --------- Launch ---------
template <typename T>
void LaunchIsNan(const T* input, bool* output, int n, musaStream_t stream) {
  if (n <= 0) return;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  IsNanKernel<T><<<blocks, threads, 0, stream>>>(input, output, n);

  // kernel 启动错误检查（和你 AddN 风格一致：错误在上层处理也行）
  musaError_t err = musaGetLastError();
  (void)err;
}

// 显式实例化
template void LaunchIsNan<float>(const float*, bool*, int, musaStream_t);
template void LaunchIsNan<double>(const double*, bool*, int, musaStream_t);
template void LaunchIsNan<Eigen::half>(const Eigen::half*, bool*, int, musaStream_t);
template void LaunchIsNan<bfloat16>(const bfloat16*, bool*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow