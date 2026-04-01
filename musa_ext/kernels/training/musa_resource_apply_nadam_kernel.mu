#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

// Fused Nadam kernel: updates var, m, v in-place
// For numeric stability and TF compatibility we use float accumulation for
// half/bfloat16 variants.

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

// Double helpers
__device__ __forceinline__ double LoadDouble(const double* p) { return *p; }
__device__ __forceinline__ void StoreDouble(double* p, double v) { *p = v; }

}  // namespace

// Kernel for float
template <typename T>
__global__ void ResourceApplyNadamKernel(T* __restrict__ var, T* __restrict__ m,
                                         T* __restrict__ v,
                                         const T beta1_power,
                                         const T beta2_power, const T lr,
                                         const T beta1, const T beta2,
                                         const T epsilon,
                                         const T* __restrict__ grad,
                                         int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // For template T=float/double, we perform typed math. For half/bf16 we will
  // use specializations that load/store via float helpers.
  T g = grad[i];
  T m_old = m[i];
  T v_old = v[i];
  T m_new = beta1 * m_old + (static_cast<T>(1.0) - beta1) * g;
  T v_new = beta2 * v_old + (static_cast<T>(1.0) - beta2) * (g * g);
  T m_bar = (beta1 * m_new + (static_cast<T>(1.0) - beta1) * g) / (static_cast<T>(1.0) - beta1_power);
  T v_hat = v_new / (static_cast<T>(1.0) - beta2_power);
  T denom = sqrt(v_hat) + epsilon;
  T var_new = var[i] - lr * m_bar / denom;
  var[i] = var_new;
  m[i] = m_new;
  v[i] = v_new;
}

template <typename Tload, typename Tstore>
__global__ void ResourceApplyNadamFloatAccumKernel(Tstore* __restrict__ var,
                                                   Tstore* __restrict__ m,
                                                   Tstore* __restrict__ v,
                                                   const float beta1_power,
                                                   const float beta2_power,
                                                   const float lr, const float beta1,
                                                   const float beta2,
                                                   const float epsilon,
                                                   const Tload* __restrict__ grad,
                                                   int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = LoadFloat(reinterpret_cast<const Tload*>(&grad[i]));
  float m_old = LoadFloat(reinterpret_cast<const Tload*>(&m[i]));
  float v_old = LoadFloat(reinterpret_cast<const Tload*>(&v[i]));
  float m_new = beta1 * m_old + (1.0f - beta1) * g;
  float v_new = beta2 * v_old + (1.0f - beta2) * (g * g);
  float m_bar = (beta1 * m_new + (1.0f - beta1) * g) / (1.0f - beta1_power);
  float v_hat = v_new / (1.0f - beta2_power);
  float denom = sqrtf(v_hat) + epsilon;
  float var_new = LoadFloat(reinterpret_cast<const Tload*>(&var[i])) - lr * m_bar / denom;
  StoreFloat(reinterpret_cast<Tstore*>(&var[i]), var_new);
  StoreFloat(reinterpret_cast<Tstore*>(&m[i]), m_new);
  StoreFloat(reinterpret_cast<Tstore*>(&v[i]), v_new);
}

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(n) (((n) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

extern "C" {
void LaunchResourceApplyNadamFloat(float* var, float* m, float* v,
                                   float beta1_power, float beta2_power,
                                   float lr, float beta1, float beta2,
                                   float epsilon, const float* grad,
                                   int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyNadamKernel<float>
      <<<OPTIMAL_BLOCKS(n), OPTIMAL_THREADS, 0, stream>>>(var, m, v,
                                                         beta1_power,
                                                         beta2_power, lr,
                                                         beta1, beta2,
                                                         epsilon, grad, n);
}

void LaunchResourceApplyNadamDouble(double* var, double* m, double* v,
                                    double beta1_power, double beta2_power,
                                    double lr, double beta1, double beta2,
                                    double epsilon, const double* grad,
                                    int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyNadamKernel<double>
      <<<OPTIMAL_BLOCKS(n), OPTIMAL_THREADS, 0, stream>>>(var, m, v,
                                                          beta1_power,
                                                          beta2_power, lr,
                                                          beta1, beta2,
                                                          epsilon, grad, n);
}

void LaunchResourceApplyNadamHalf(void* var, void* m, void* v,
                                  float beta1_power, float beta2_power,
                                  float lr, float beta1, float beta2,
                                  float epsilon, const void* grad,
                                  int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyNadamFloatAccumKernel<Eigen::half, Eigen::half>
      <<<OPTIMAL_BLOCKS(n), OPTIMAL_THREADS, 0, stream>>>(
          reinterpret_cast<Eigen::half*>(var), reinterpret_cast<Eigen::half*>(m),
          reinterpret_cast<Eigen::half*>(v), beta1_power, beta2_power, lr,
          beta1, beta2, epsilon, reinterpret_cast<const Eigen::half*>(grad), n);
}

void LaunchResourceApplyNadamBFloat16(void* var, void* m, void* v,
                                      float beta1_power, float beta2_power,
                                      float lr, float beta1, float beta2,
                                      float epsilon, const void* grad,
                                      int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  ResourceApplyNadamFloatAccumKernel<bfloat16, bfloat16>
      <<<OPTIMAL_BLOCKS(n), OPTIMAL_THREADS, 0, stream>>>(
          reinterpret_cast<bfloat16*>(var), reinterpret_cast<bfloat16*>(m),
          reinterpret_cast<bfloat16*>(v), beta1_power, beta2_power, lr,
          beta1, beta2, epsilon, reinterpret_cast<const bfloat16*>(grad), n);
}
}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
