#include <musa_runtime.h>
#include <musa_fp16.h>
#include <math.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

using bfloat16 = tensorflow::bfloat16;

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

__device__ __forceinline__ void StoreFloat(float* p, float v) {
    *p = v;
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
    __half h = __float2half(v);
    *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
    uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
    uint16_t b_val = *f_ptr >> 16;
    *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__global__ void SigmoidCalibrationKernel(const T* input, const T* scale,
                                         T* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = LoadFloat(&input[idx]);
    float s = LoadFloat(&scale[idx]);
    float result = x / (x + s * (1.0f - x));
    StoreFloat(&output[idx], result);
  }
}

template <typename T>
void LaunchSigmoidCalibrationKernel(const void* input, const void* scale,
                                    void* output, int n, musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  SigmoidCalibrationKernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(scale),
      static_cast<T*>(output), n);
}

template void LaunchSigmoidCalibrationKernel<float>(const void*, const void*,
                                                   void*, int, musaStream_t);
template void LaunchSigmoidCalibrationKernel<Eigen::half>(const void*,
                                                         const void*, void*,
                                                         int, musaStream_t);
template void LaunchSigmoidCalibrationKernel<bfloat16>(const void*, const void*,
                                                      void*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
