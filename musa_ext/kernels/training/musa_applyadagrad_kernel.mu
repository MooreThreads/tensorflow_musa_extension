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

template <typename T, bool UpdateSlots>
__global__ void FusedAdagradV2Kernel(T* __restrict__ var,
                                     T* __restrict__ accum,
                                     const T* __restrict__ grad, double lr,
                                     double epsilon, int64_t n) {
  using AccT = decltype(LoadValue(var));
  const int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);

  const AccT lr_v = static_cast<AccT>(lr);
  const AccT epsilon_v = static_cast<AccT>(epsilon);

  for (int64_t i = tid; i < n; i += stride) {
    const AccT g = LoadValue(&grad[i]);
    const AccT accum_old = LoadValue(&accum[i]);
    const AccT accum_new = UpdateSlots ? accum_old + g * g : accum_old;
    const AccT var_new =
        LoadValue(&var[i]) - lr_v * g / (DeviceSqrt(accum_new) + epsilon_v);

    if (UpdateSlots) {
      StoreValue(&accum[i], accum_new);
    }
    StoreValue(&var[i], var_new);
  }
}

template <bool UpdateSlots>
__device__ __forceinline__ void UpdateAdagradLane(float* var, float* accum,
                                                  float grad, float lr,
                                                  float epsilon) {
  const float accum_old = *accum;
  const float accum_new = UpdateSlots ? accum_old + grad * grad : accum_old;
  *var -= lr * grad / (sqrtf(accum_new) + epsilon);
  if (UpdateSlots) {
    *accum = accum_new;
  }
}

template <bool UpdateSlots>
__global__ void FusedAdagradV2Float4Kernel(float* __restrict__ var,
                                           float* __restrict__ accum,
                                           const float* __restrict__ grad,
                                           float lr, float epsilon,
                                           int64_t n4) {
  const int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride = gridDim.x * static_cast<int64_t>(blockDim.x);

  float4* var4 = reinterpret_cast<float4*>(var);
  float4* accum4 = reinterpret_cast<float4*>(accum);
  const float4* grad4 = reinterpret_cast<const float4*>(grad);

  for (int64_t i = tid; i < n4; i += stride) {
    float4 v_var = var4[i];
    float4 v_accum = accum4[i];
    const float4 v_grad = grad4[i];

    UpdateAdagradLane<UpdateSlots>(&v_var.x, &v_accum.x, v_grad.x, lr,
                                   epsilon);
    UpdateAdagradLane<UpdateSlots>(&v_var.y, &v_accum.y, v_grad.y, lr,
                                   epsilon);
    UpdateAdagradLane<UpdateSlots>(&v_var.z, &v_accum.z, v_grad.z, lr,
                                   epsilon);
    UpdateAdagradLane<UpdateSlots>(&v_var.w, &v_accum.w, v_grad.w, lr,
                                   epsilon);

    if (UpdateSlots) {
      accum4[i] = v_accum;
    }
    var4[i] = v_var;
  }
}

inline int BlocksFor(int64_t work_items, int threads) {
  int64_t blocks = (work_items + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}

template <typename T, bool UpdateSlots>
void LaunchAdagradV2Scalar(T* var, T* accum, const T* grad, double lr,
                           double epsilon, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  if (n <= kSmallElements) {
    FusedAdagradV2Kernel<T, UpdateSlots><<<1, kSmallThreads, 0, stream>>>(
        var, accum, grad, lr, epsilon, n);
    return;
  }

  FusedAdagradV2Kernel<T, UpdateSlots>
      <<<BlocksFor(n, kDefaultThreads), kDefaultThreads, 0, stream>>>(
          var, accum, grad, lr, epsilon, n);
}

inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & static_cast<uintptr_t>(0xF)) == 0;
}

}  // namespace

template <typename T>
void LaunchFusedApplyAdagradV2Kernel(T* var, T* accum, const T* grad,
                                     double lr, double epsilon,
                                     bool update_slots, int64_t n,
                                     musaStream_t stream) {
  if (update_slots) {
    LaunchAdagradV2Scalar<T, true>(var, accum, grad, lr, epsilon, n, stream);
  } else {
    LaunchAdagradV2Scalar<T, false>(var, accum, grad, lr, epsilon, n, stream);
  }
}

template <>
void LaunchFusedApplyAdagradV2Kernel<float>(float* var, float* accum,
                                            const float* grad, double lr,
                                            double epsilon, bool update_slots,
                                            int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  if (n >= kVectorizedMinElements && (n % 4) == 0 && IsAligned16(var) &&
      IsAligned16(accum) && IsAligned16(grad)) {
    const int64_t n4 = n / 4;
    if (update_slots) {
      FusedAdagradV2Float4Kernel<true>
          <<<BlocksFor(n4, kDefaultThreads), kDefaultThreads, 0, stream>>>(
              var, accum, grad, static_cast<float>(lr),
              static_cast<float>(epsilon), n4);
    } else {
      FusedAdagradV2Float4Kernel<false>
          <<<BlocksFor(n4, kDefaultThreads), kDefaultThreads, 0, stream>>>(
              var, accum, grad, static_cast<float>(lr),
              static_cast<float>(epsilon), n4);
    }
    return;
  }

  if (update_slots) {
    LaunchAdagradV2Scalar<float, true>(var, accum, grad, lr, epsilon, n,
                                       stream);
  } else {
    LaunchAdagradV2Scalar<float, false>(var, accum, grad, lr, epsilon, n,
                                        stream);
  }
}

#define REGISTER_FUSED_ADAGRAD_V2_LAUNCHER(T)                         \
  template void LaunchFusedApplyAdagradV2Kernel<T>(                    \
      T * var, T * accum, const T* grad, double lr, double epsilon,    \
      bool update_slots, int64_t n, musaStream_t stream);

REGISTER_FUSED_ADAGRAD_V2_LAUNCHER(double);
REGISTER_FUSED_ADAGRAD_V2_LAUNCHER(Eigen::half);
REGISTER_FUSED_ADAGRAD_V2_LAUNCHER(bfloat16);

#undef REGISTER_FUSED_ADAGRAD_V2_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
