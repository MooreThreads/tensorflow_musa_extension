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

// Mixed-precision ApplyAdam kernel.
//
// State (var/m/v) is always fp32. The gradient is loaded as fp32, fp16, or
// bf16 and promoted to fp32 in registers using the SDK's RNE intrinsics, so
// per-element promotion error is at most 0.5 ULP and the Adam update itself
// runs entirely in fp32. This avoids both:
//   * the precision loss of doing the math in bf16 (the legacy
//     MusaResourceApplyAdam<bfloat16> path), and
//   * the extra device-memory round-trip of an explicit Cast(bf16 -> fp32)
//     in front of a fp32 ApplyAdam.

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

__device__ __forceinline__ float LoadGradAsFloat(const float* p) { return *p; }

__device__ __forceinline__ float LoadGradAsFloat(const Eigen::half* p) {
  const __half h = *reinterpret_cast<const __half*>(p);
  return __half2float(h);
}

__device__ __forceinline__ float LoadGradAsFloat(const bfloat16* p) {
  const __mt_bfloat16 b = *reinterpret_cast<const __mt_bfloat16*>(p);
  return __bfloat162float(b);
}

template <typename GradT>
__global__ __launch_bounds__(256) void ApplyAdamMixedKernel(
    float* __restrict__ var, float* __restrict__ m, float* __restrict__ v,
    const GradT* __restrict__ grad, float lr_t, float beta1, float one_minus_b1,
    float beta2, float one_minus_b2, float epsilon, int64_t n,
    bool use_nesterov) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  const float g = LoadGradAsFloat(grad + idx);
  const float m_val = m[idx];
  const float v_val = v[idx];
  const float var_val = var[idx];

  // m_t = beta1 * m + (1 - beta1) * g
  const float m_new = beta1 * m_val + one_minus_b1 * g;
  // v_t = beta2 * v + (1 - beta2) * g^2
  const float v_new = beta2 * v_val + one_minus_b2 * g * g;

  // Optional Nesterov-style numerator. The legacy MusaResourceApplyAdam path
  // silently ignores use_nesterov; for the mixed-precision path we honor it
  // because Keras' Nadam pipeline (and any user who explicitly sets the attr)
  // would otherwise get standard Adam behavior masked behind a different op
  // name.
  const float numer = use_nesterov ? (beta1 * m_new + one_minus_b1 * g) : m_new;
  const float var_new = var_val - lr_t * numer / (sqrtf(v_new) + epsilon);

  m[idx] = m_new;
  v[idx] = v_new;
  var[idx] = var_new;
}

template <typename GradT>
void LaunchApplyAdamMixedImpl(float* var, float* m, float* v, const GradT* grad,
                              float lr_t, float beta1, float beta2,
                              float epsilon, int64_t n, bool use_nesterov,
                              musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  // Precompute (1 - beta) on the host so the per-thread inner loop stays at
  // 3 FMA-shaped ops.
  const float one_minus_b1 = 1.0f - beta1;
  const float one_minus_b2 = 1.0f - beta2;
  ApplyAdamMixedKernel<GradT><<<blocks, kThreads, 0, stream>>>(
      var, m, v, grad, lr_t, beta1, one_minus_b1, beta2, one_minus_b2, epsilon,
      n, use_nesterov);
}

}  // namespace

// Plain-C launchers so the C++ op file does not have to instantiate a kernel
// template across .cc / .mu boundaries.
extern "C" {

void LaunchApplyAdamMixed_Float(float* var, float* m, float* v,
                                const void* grad, float lr_t, float beta1,
                                float beta2, float epsilon, int64_t n,
                                bool use_nesterov, musaStream_t stream) {
  LaunchApplyAdamMixedImpl<float>(var, m, v,
                                  reinterpret_cast<const float*>(grad), lr_t,
                                  beta1, beta2, epsilon, n, use_nesterov,
                                  stream);
}

void LaunchApplyAdamMixed_Half(float* var, float* m, float* v,
                               const void* grad, float lr_t, float beta1,
                               float beta2, float epsilon, int64_t n,
                               bool use_nesterov, musaStream_t stream) {
  LaunchApplyAdamMixedImpl<Eigen::half>(
      var, m, v, reinterpret_cast<const Eigen::half*>(grad), lr_t, beta1, beta2,
      epsilon, n, use_nesterov, stream);
}

void LaunchApplyAdamMixed_BFloat16(float* var, float* m, float* v,
                                   const void* grad, float lr_t, float beta1,
                                   float beta2, float epsilon, int64_t n,
                                   bool use_nesterov, musaStream_t stream) {
  LaunchApplyAdamMixedImpl<bfloat16>(
      var, m, v, reinterpret_cast<const bfloat16*>(grad), lr_t, beta1, beta2,
      epsilon, n, use_nesterov, stream);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
