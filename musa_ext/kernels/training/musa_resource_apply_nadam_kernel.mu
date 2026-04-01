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

namespace {
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

__device__ __forceinline__ double LoadFloat(const double* p) { return *p; }
__device__ __forceinline__ void StoreFloat(double* p, double v) { *p = v; }
}  // namespace

template <typename T>
__global__ void ResourceApplyNadamKernel(T* __restrict__ var, T* __restrict__ m,
                                         T* __restrict__ v,
                                         const T* __restrict__ grad,
                                         float beta1_power, float beta2_power,
                                         float lr, float beta1, float beta2,
                                         float epsilon, int64_t n) {
  const int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (tid >= n) return;

  float grad_val = LoadFloat(&grad[tid]);
  float m_val = LoadFloat(&m[tid]);
  float v_val = LoadFloat(&v[tid]);
  float var_val = LoadFloat(&var[tid]);

  // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
  m_val = beta1 * m_val + (1.0f - beta1) * grad_val;
  // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
  v_val = beta2 * v_val + (1.0f - beta2) * grad_val * grad_val;

  // m_hat = (beta1 * m_val + (1 - beta1) * grad_val) / (1 - beta1_power)
  float m_hat =
      (beta1 * m_val + (1.0f - beta1) * grad_val) / (1.0f - beta1_power);
  // v_hat = v_t / (1 - beta2_power)
  float v_hat = v_val / (1.0f - beta2_power);

  // var_t = var_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
  var_val -= lr * m_hat / (sqrtf(v_hat) + epsilon);

  StoreFloat(&m[tid], m_val);
  StoreFloat(&v[tid], v_val);
  StoreFloat(&var[tid], var_val);
}

template <typename T>
void LaunchResourceApplyNadamKernel(T* var, T* m, T* v, const T* grad,
                                    float beta1_power, float beta2_power,
                                    float lr, float beta1, float beta2,
                                    float epsilon, int64_t n,
                                    musaStream_t stream) {
  if (n <= 0) return;
  int block_size = 256;
  int64_t num_blocks = (n + block_size - 1) / block_size;
  ResourceApplyNadamKernel<T><<<num_blocks, block_size, 0, stream>>>(
      var, m, v, grad, beta1_power, beta2_power, lr, beta1, beta2, epsilon, n);
}

#define REGISTER_NADAM_LAUNCHER(T)                                          \
  template void LaunchResourceApplyNadamKernel<T>(                          \
      T * var, T * m, T * v, const T* grad, float beta1_power,              \
      float beta2_power, float lr, float beta1, float beta2, float epsilon, \
      int64_t n, musaStream_t stream);

REGISTER_NADAM_LAUNCHER(float);
REGISTER_NADAM_LAUNCHER(Eigen::half);
REGISTER_NADAM_LAUNCHER(bfloat16);
REGISTER_NADAM_LAUNCHER(double);

#undef REGISTER_NADAM_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
