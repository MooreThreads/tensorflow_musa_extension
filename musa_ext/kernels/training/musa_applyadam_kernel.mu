/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <math.h>
#include <musa_bf16.h>
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
constexpr int kSmallThreads = 128;
constexpr int kDefaultThreads = 512;
constexpr int kMaxBlocks = 8192;
constexpr int64_t kSmallElements = 1024;
constexpr int64_t kVectorizedMinElements = 4096;
__device__ __forceinline__ float LoadValue(const float* p) { return *p; }
__device__ __forceinline__ void StoreValue(float* p, float v) { *p = v; }
__device__ __forceinline__ double LoadValue(const double* p) { return *p; }
__device__ __forceinline__ void StoreValue(double* p, double v) { *p = v; }
__device__ __forceinline__ float LoadValue(const Eigen::half* p) {
  const __half h = *reinterpret_cast<const __half*>(p);
  return __half2float(h);
}
__device__ __forceinline__ void StoreValue(Eigen::half* p, float v) {
  const __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}
__device__ __forceinline__ float LoadValue(const bfloat16* p) {
  const __mt_bfloat16 b = *reinterpret_cast<const __mt_bfloat16*>(p);
  return __bfloat162float(b);
}
__device__ __forceinline__ void StoreValue(bfloat16* p, float v) {
  const __mt_bfloat16 b = __float2bfloat16(v);
  *reinterpret_cast<__mt_bfloat16*>(p) = b;
}
__device__ __forceinline__ float DeviceSqrt(float v) { return sqrtf(v); }
__device__ __forceinline__ double DeviceSqrt(double v) { return sqrt(v); }
template <typename AccT, bool UseNesterov>
__device__ __forceinline__ void AdamUpdate(AccT var_old, AccT m_old,
                                           AccT v_old, AccT g, AccT alpha,
                                           AccT beta1, AccT one_minus_beta1,
                                           AccT beta2, AccT one_minus_beta2,
                                           AccT epsilon, AccT* var_new,
                                           AccT* m_new, AccT* v_new) {
  *m_new = beta1 * m_old + one_minus_beta1 * g;
  *v_new = beta2 * v_old + one_minus_beta2 * g * g;
  const AccT numerator =
      UseNesterov ? beta1 * (*m_new) + one_minus_beta1 * g : *m_new;
  *var_new = var_old - alpha * numerator / (DeviceSqrt(*v_new) + epsilon);
}
template <typename T, bool UseNesterov>
__global__ void FusedAdamKernel(T* __restrict__ var, T* __restrict__ m,
                                T* __restrict__ v,
                                const T* __restrict__ grad, double alpha,
                                double beta1, double beta2, double epsilon,
                                int64_t n) {
  using AccT = decltype(LoadValue(var));
  const int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);
  const AccT alpha_v = static_cast<AccT>(alpha);
  const AccT beta1_v = static_cast<AccT>(beta1);
  const AccT beta2_v = static_cast<AccT>(beta2);
  const AccT one_minus_beta1_v = static_cast<AccT>(1.0) - beta1_v;
  const AccT one_minus_beta2_v = static_cast<AccT>(1.0) - beta2_v;
  const AccT epsilon_v = static_cast<AccT>(epsilon);
  for (int64_t i = tid; i < n; i += stride) {
    AccT var_new;
    AccT m_new;
    AccT v_new;
    AdamUpdate<AccT, UseNesterov>(
        LoadValue(&var[i]), LoadValue(&m[i]), LoadValue(&v[i]),
        LoadValue(&grad[i]), alpha_v, beta1_v, one_minus_beta1_v, beta2_v,
        one_minus_beta2_v, epsilon_v, &var_new, &m_new, &v_new);
    StoreValue(&m[i], m_new);
    StoreValue(&v[i], v_new);
    StoreValue(&var[i], var_new);
  }
}
template <bool UseNesterov>
__device__ __forceinline__ void AdamUpdateFloatLane(
    float* var, float* m, float* v, float grad, float alpha, float beta1,
    float one_minus_beta1, float beta2, float one_minus_beta2, float epsilon) {
  float var_new;
  float m_new;
  float v_new;
  AdamUpdate<float, UseNesterov>(*var, *m, *v, grad, alpha, beta1,
                                 one_minus_beta1, beta2, one_minus_beta2,
                                 epsilon, &var_new, &m_new, &v_new);
  *m = m_new;
  *v = v_new;
  *var = var_new;
}
template <bool UseNesterov>
__global__ void FusedAdamFloat4Kernel(float* __restrict__ var,
                                      float* __restrict__ m,
                                      float* __restrict__ v,
                                      const float* __restrict__ grad,
                                      float alpha, float beta1, float beta2,
                                      float epsilon, int64_t n4) {
  const int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);
  const float one_minus_beta1 = 1.0f - beta1;
  const float one_minus_beta2 = 1.0f - beta2;
  float4* var4 = reinterpret_cast<float4*>(var);
  float4* m4 = reinterpret_cast<float4*>(m);
  float4* v4 = reinterpret_cast<float4*>(v);
  const float4* grad4 = reinterpret_cast<const float4*>(grad);
  for (int64_t i = tid; i < n4; i += stride) {
    float4 var_val = var4[i];
    float4 m_val = m4[i];
    float4 v_val = v4[i];
    const float4 grad_val = grad4[i];
    AdamUpdateFloatLane<UseNesterov>(&var_val.x, &m_val.x, &v_val.x,
                                     grad_val.x, alpha, beta1,
                                     one_minus_beta1, beta2, one_minus_beta2,
                                     epsilon);
    AdamUpdateFloatLane<UseNesterov>(&var_val.y, &m_val.y, &v_val.y,
                                     grad_val.y, alpha, beta1,
                                     one_minus_beta1, beta2, one_minus_beta2,
                                     epsilon);
    AdamUpdateFloatLane<UseNesterov>(&var_val.z, &m_val.z, &v_val.z,
                                     grad_val.z, alpha, beta1,
                                     one_minus_beta1, beta2, one_minus_beta2,
                                     epsilon);
    AdamUpdateFloatLane<UseNesterov>(&var_val.w, &m_val.w, &v_val.w,
                                     grad_val.w, alpha, beta1,
                                     one_minus_beta1, beta2, one_minus_beta2,
                                     epsilon);
    m4[i] = m_val;
    v4[i] = v_val;
    var4[i] = var_val;
  }
}
inline int BlocksFor(int64_t work_items, int threads) {
  int64_t blocks = (work_items + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}
inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & static_cast<uintptr_t>(0xF)) == 0;
}
template <typename T, bool UseNesterov>
void LaunchAdamScalar(T* var, T* m, T* v, const T* grad, double alpha,
                      double beta1, double beta2, double epsilon, int64_t n,
                      musaStream_t stream) {
  if (n <= 0) return;
  if (n <= kSmallElements) {
    FusedAdamKernel<T, UseNesterov><<<1, kSmallThreads, 0, stream>>>(
        var, m, v, grad, alpha, beta1, beta2, epsilon, n);
    return;
  }
  FusedAdamKernel<T, UseNesterov>
      <<<BlocksFor(n, kDefaultThreads), kDefaultThreads, 0, stream>>>(
          var, m, v, grad, alpha, beta1, beta2, epsilon, n);
}
}  // namespace
template <typename T>
void LaunchFusedApplyAdamKernel(T* var, T* m, T* v, const T* grad,
                                double alpha, double beta1, double beta2,
                                double epsilon, bool use_nesterov, int64_t n,
                                musaStream_t stream) {
  if (use_nesterov) {
    LaunchAdamScalar<T, true>(var, m, v, grad, alpha, beta1, beta2, epsilon, n,
                              stream);
  } else {
    LaunchAdamScalar<T, false>(var, m, v, grad, alpha, beta1, beta2, epsilon,
                               n, stream);
  }
}
template <>
void LaunchFusedApplyAdamKernel<float>(
    float* var, float* m, float* v, const float* grad, double alpha,
    double beta1, double beta2, double epsilon, bool use_nesterov, int64_t n,
    musaStream_t stream) {
  if (n <= 0) return;
  if (n >= kVectorizedMinElements && (n % 4) == 0 && IsAligned16(var) &&
      IsAligned16(m) && IsAligned16(v) && IsAligned16(grad)) {
    const int64_t n4 = n / 4;
    if (use_nesterov) {
      FusedAdamFloat4Kernel<true>
          <<<BlocksFor(n4, kDefaultThreads), kDefaultThreads, 0, stream>>>(
              var, m, v, grad, static_cast<float>(alpha),
              static_cast<float>(beta1), static_cast<float>(beta2),
              static_cast<float>(epsilon), n4);
    } else {
      FusedAdamFloat4Kernel<false>
          <<<BlocksFor(n4, kDefaultThreads), kDefaultThreads, 0, stream>>>(
              var, m, v, grad, static_cast<float>(alpha),
              static_cast<float>(beta1), static_cast<float>(beta2),
              static_cast<float>(epsilon), n4);
    }
    return;
  }
  if (use_nesterov) {
    LaunchAdamScalar<float, true>(var, m, v, grad, alpha, beta1, beta2,
                                  epsilon, n, stream);
  } else {
    LaunchAdamScalar<float, false>(var, m, v, grad, alpha, beta1, beta2,
                                   epsilon, n, stream);
  }
}
#define REGISTER_FUSED_ADAM_LAUNCHER(T)                                  \
  template void LaunchFusedApplyAdamKernel<T>(                            \
      T * var, T * m, T * v, const T* grad, double alpha, double beta1,   \
      double beta2, double epsilon, bool use_nesterov, int64_t n,         \
      musaStream_t stream);
REGISTER_FUSED_ADAM_LAUNCHER(double);
REGISTER_FUSED_ADAM_LAUNCHER(Eigen::half);
REGISTER_FUSED_ADAM_LAUNCHER(bfloat16);
#undef REGISTER_FUSED_ADAM_LAUNCHER
}  // namespace musa
}  // namespace tensorflow