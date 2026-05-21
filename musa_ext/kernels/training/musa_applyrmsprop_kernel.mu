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
constexpr int kDefaultThreads = 256;
constexpr int kMaxBlocks = 4096;
constexpr int64_t kSmallElements = 1024;
constexpr int64_t kVectorizedMinElements = 4096;

__device__ __forceinline__ float LoadValue(const float* p) { return *p; }

__device__ __forceinline__ void StoreValue(float* p, float v) { *p = v; }

__device__ __forceinline__ double LoadValue(const double* p) { return *p; }

__device__ __forceinline__ void StoreValue(double* p, double v) { *p = v; }

__device__ __forceinline__ float LoadValue(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreValue(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadValue(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreValue(bfloat16* p, float v) {
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
  uint16_t b_val = static_cast<uint16_t>(*f_ptr >> 16);
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

__device__ __forceinline__ float DeviceSqrt(float v) { return sqrtf(v); }

__device__ __forceinline__ double DeviceSqrt(double v) { return sqrt(v); }

template <typename T>
__global__ void FusedRMSPropKernel(T* __restrict__ var, T* __restrict__ ms,
                                   T* __restrict__ mom,
                                   const T* __restrict__ grad, double lr,
                                   double rho, double momentum, double epsilon,
                                   int64_t n) {
  using AccT = decltype(LoadValue(var));
  const int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);

  const AccT lr_v = static_cast<AccT>(lr);
  const AccT rho_v = static_cast<AccT>(rho);
  const AccT one_minus_rho_v = static_cast<AccT>(1.0) - rho_v;
  const AccT momentum_v = static_cast<AccT>(momentum);
  const AccT epsilon_v = static_cast<AccT>(epsilon);

  for (int64_t i = tid; i < n; i += stride) {
    const AccT g = LoadValue(&grad[i]);
    const AccT ms_new = rho_v * LoadValue(&ms[i]) + one_minus_rho_v * g * g;
    const AccT mom_new =
        momentum_v * LoadValue(&mom[i]) + lr_v * g / DeviceSqrt(ms_new + epsilon_v);
    const AccT var_new = LoadValue(&var[i]) - mom_new;

    StoreValue(&ms[i], ms_new);
    StoreValue(&mom[i], mom_new);
    StoreValue(&var[i], var_new);
  }
}

template <typename T>
__global__ void FusedCenteredRMSPropKernel(
    T* __restrict__ var, T* __restrict__ mg, T* __restrict__ ms,
    T* __restrict__ mom, const T* __restrict__ grad, double lr, double rho,
    double momentum, double epsilon, int64_t n) {
  using AccT = decltype(LoadValue(var));
  const int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);

  const AccT lr_v = static_cast<AccT>(lr);
  const AccT rho_v = static_cast<AccT>(rho);
  const AccT one_minus_rho_v = static_cast<AccT>(1.0) - rho_v;
  const AccT momentum_v = static_cast<AccT>(momentum);
  const AccT epsilon_v = static_cast<AccT>(epsilon);

  for (int64_t i = tid; i < n; i += stride) {
    const AccT g = LoadValue(&grad[i]);
    const AccT mg_new = rho_v * LoadValue(&mg[i]) + one_minus_rho_v * g;
    const AccT ms_new = rho_v * LoadValue(&ms[i]) + one_minus_rho_v * g * g;
    const AccT denom = ms_new - mg_new * mg_new + epsilon_v;
    const AccT mom_new =
        momentum_v * LoadValue(&mom[i]) + lr_v * g / DeviceSqrt(denom);
    const AccT var_new = LoadValue(&var[i]) - mom_new;

    StoreValue(&mg[i], mg_new);
    StoreValue(&ms[i], ms_new);
    StoreValue(&mom[i], mom_new);
    StoreValue(&var[i], var_new);
  }
}

__device__ __forceinline__ float RMSPropMom(float ms_old, float mom_old,
                                            float g, float lr, float rho,
                                            float momentum, float epsilon,
                                            float* ms_new) {
  const float one_minus_rho = 1.0f - rho;
  *ms_new = rho * ms_old + one_minus_rho * g * g;
  return momentum * mom_old + lr * g / sqrtf(*ms_new + epsilon);
}

__global__ void FusedRMSPropFloat4Kernel(
    float* __restrict__ var, float* __restrict__ ms, float* __restrict__ mom,
    const float* __restrict__ grad, float lr, float rho, float momentum,
    float epsilon, int64_t n4) {
  const int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);

  float4* var4 = reinterpret_cast<float4*>(var);
  float4* ms4 = reinterpret_cast<float4*>(ms);
  float4* mom4 = reinterpret_cast<float4*>(mom);
  const float4* grad4 = reinterpret_cast<const float4*>(grad);

  for (int64_t i = tid; i < n4; i += stride) {
    float4 v_var = var4[i];
    float4 v_ms = ms4[i];
    float4 v_mom = mom4[i];
    const float4 v_grad = grad4[i];

    float ms_new;
    float mom_new;

    mom_new = RMSPropMom(v_ms.x, v_mom.x, v_grad.x, lr, rho, momentum, epsilon,
                         &ms_new);
    v_ms.x = ms_new;
    v_mom.x = mom_new;
    v_var.x -= mom_new;

    mom_new = RMSPropMom(v_ms.y, v_mom.y, v_grad.y, lr, rho, momentum, epsilon,
                         &ms_new);
    v_ms.y = ms_new;
    v_mom.y = mom_new;
    v_var.y -= mom_new;

    mom_new = RMSPropMom(v_ms.z, v_mom.z, v_grad.z, lr, rho, momentum, epsilon,
                         &ms_new);
    v_ms.z = ms_new;
    v_mom.z = mom_new;
    v_var.z -= mom_new;

    mom_new = RMSPropMom(v_ms.w, v_mom.w, v_grad.w, lr, rho, momentum, epsilon,
                         &ms_new);
    v_ms.w = ms_new;
    v_mom.w = mom_new;
    v_var.w -= mom_new;

    ms4[i] = v_ms;
    mom4[i] = v_mom;
    var4[i] = v_var;
  }
}

__global__ void FusedCenteredRMSPropFloat4Kernel(
    float* __restrict__ var, float* __restrict__ mg, float* __restrict__ ms,
    float* __restrict__ mom, const float* __restrict__ grad, float lr,
    float rho, float momentum, float epsilon, int64_t n4) {
  const int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);
  const float one_minus_rho = 1.0f - rho;

  float4* var4 = reinterpret_cast<float4*>(var);
  float4* mg4 = reinterpret_cast<float4*>(mg);
  float4* ms4 = reinterpret_cast<float4*>(ms);
  float4* mom4 = reinterpret_cast<float4*>(mom);
  const float4* grad4 = reinterpret_cast<const float4*>(grad);

  for (int64_t i = tid; i < n4; i += stride) {
    float4 v_var = var4[i];
    float4 v_mg = mg4[i];
    float4 v_ms = ms4[i];
    float4 v_mom = mom4[i];
    const float4 v_grad = grad4[i];

#define UPDATE_CENTERED_LANE(lane)                                             \
    do {                                                                       \
      const float g = v_grad.lane;                                             \
      const float mg_new = rho * v_mg.lane + one_minus_rho * g;                \
      const float ms_new = rho * v_ms.lane + one_minus_rho * g * g;            \
      const float mom_new =                                                     \
          momentum * v_mom.lane + lr * g / sqrtf(ms_new - mg_new * mg_new + epsilon); \
      v_mg.lane = mg_new;                                                      \
      v_ms.lane = ms_new;                                                      \
      v_mom.lane = mom_new;                                                    \
      v_var.lane -= mom_new;                                                   \
    } while (0)

    UPDATE_CENTERED_LANE(x);
    UPDATE_CENTERED_LANE(y);
    UPDATE_CENTERED_LANE(z);
    UPDATE_CENTERED_LANE(w);

#undef UPDATE_CENTERED_LANE

    mg4[i] = v_mg;
    ms4[i] = v_ms;
    mom4[i] = v_mom;
    var4[i] = v_var;
  }
}

inline int BlocksFor(int64_t work_items, int threads) {
  int64_t blocks = (work_items + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}

template <typename T>
void LaunchRMSPropScalar(T* var, T* ms, T* mom, const T* grad, double lr,
                         double rho, double momentum, double epsilon,
                         int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  if (n <= kSmallElements) {
    FusedRMSPropKernel<T><<<1, kSmallThreads, 0, stream>>>(
        var, ms, mom, grad, lr, rho, momentum, epsilon, n);
    return;
  }
  FusedRMSPropKernel<T><<<BlocksFor(n, kDefaultThreads), kDefaultThreads, 0,
                         stream>>>(var, ms, mom, grad, lr, rho, momentum,
                                    epsilon, n);
}

template <typename T>
void LaunchCenteredRMSPropScalar(T* var, T* mg, T* ms, T* mom, const T* grad,
                                 double lr, double rho, double momentum,
                                 double epsilon, int64_t n,
                                 musaStream_t stream) {
  if (n <= 0) return;
  if (n <= kSmallElements) {
    FusedCenteredRMSPropKernel<T><<<1, kSmallThreads, 0, stream>>>(
        var, mg, ms, mom, grad, lr, rho, momentum, epsilon, n);
    return;
  }
  FusedCenteredRMSPropKernel<T><<<BlocksFor(n, kDefaultThreads),
                                 kDefaultThreads, 0, stream>>>(
      var, mg, ms, mom, grad, lr, rho, momentum, epsilon, n);
}

inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & static_cast<uintptr_t>(0xF)) == 0;
}

}  // namespace

template <typename T>
void LaunchFusedResourceApplyRMSPropKernel(
    T* var, T* ms, T* mom, const T* grad, double lr, double rho,
    double momentum, double epsilon, int64_t n, musaStream_t stream) {
  LaunchRMSPropScalar(var, ms, mom, grad, lr, rho, momentum, epsilon, n, stream);
}

template <>
void LaunchFusedResourceApplyRMSPropKernel<float>(
    float* var, float* ms, float* mom, const float* grad, double lr, double rho,
    double momentum, double epsilon, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  if (n >= kVectorizedMinElements && (n % 4) == 0 && IsAligned16(var) &&
      IsAligned16(ms) && IsAligned16(mom) && IsAligned16(grad)) {
    const int64_t n4 = n / 4;
    FusedRMSPropFloat4Kernel<<<BlocksFor(n4, kDefaultThreads),
                               kDefaultThreads, 0, stream>>>(
        var, ms, mom, grad, static_cast<float>(lr), static_cast<float>(rho),
        static_cast<float>(momentum), static_cast<float>(epsilon), n4);
    return;
  }
  LaunchRMSPropScalar(var, ms, mom, grad, lr, rho, momentum, epsilon, n, stream);
}

template <typename T>
void LaunchFusedResourceApplyCenteredRMSPropKernel(
    T* var, T* mg, T* ms, T* mom, const T* grad, double lr, double rho,
    double momentum, double epsilon, int64_t n, musaStream_t stream) {
  LaunchCenteredRMSPropScalar(var, mg, ms, mom, grad, lr, rho, momentum, epsilon,
                              n, stream);
}

template <>
void LaunchFusedResourceApplyCenteredRMSPropKernel<float>(
    float* var, float* mg, float* ms, float* mom, const float* grad, double lr,
    double rho, double momentum, double epsilon, int64_t n,
    musaStream_t stream) {
  if (n <= 0) return;
  if (n >= kVectorizedMinElements && (n % 4) == 0 && IsAligned16(var) &&
      IsAligned16(mg) && IsAligned16(ms) && IsAligned16(mom) &&
      IsAligned16(grad)) {
    const int64_t n4 = n / 4;
    FusedCenteredRMSPropFloat4Kernel<<<BlocksFor(n4, kDefaultThreads),
                                       kDefaultThreads, 0, stream>>>(
        var, mg, ms, mom, grad, static_cast<float>(lr),
        static_cast<float>(rho), static_cast<float>(momentum),
        static_cast<float>(epsilon), n4);
    return;
  }
  LaunchCenteredRMSPropScalar(var, mg, ms, mom, grad, lr, rho, momentum, epsilon,
                              n, stream);
}

#define REGISTER_FUSED_RMSPROP_LAUNCHER(T)                                \
  template void LaunchFusedResourceApplyRMSPropKernel<T>(                  \
      T * var, T * ms, T * mom, const T* grad, double lr, double rho,      \
      double momentum, double epsilon, int64_t n, musaStream_t stream);

#define REGISTER_FUSED_CENTERED_RMSPROP_LAUNCHER(T)                       \
  template void LaunchFusedResourceApplyCenteredRMSPropKernel<T>(          \
      T * var, T * mg, T * ms, T * mom, const T* grad, double lr,          \
      double rho, double momentum, double epsilon, int64_t n,              \
      musaStream_t stream);

REGISTER_FUSED_RMSPROP_LAUNCHER(double);
REGISTER_FUSED_RMSPROP_LAUNCHER(Eigen::half);
REGISTER_FUSED_RMSPROP_LAUNCHER(bfloat16);

REGISTER_FUSED_CENTERED_RMSPROP_LAUNCHER(double);
REGISTER_FUSED_CENTERED_RMSPROP_LAUNCHER(Eigen::half);
REGISTER_FUSED_CENTERED_RMSPROP_LAUNCHER(bfloat16);

#undef REGISTER_FUSED_RMSPROP_LAUNCHER
#undef REGISTER_FUSED_CENTERED_RMSPROP_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
