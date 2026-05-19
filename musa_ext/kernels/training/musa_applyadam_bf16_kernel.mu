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

// FP32-internal-compute ApplyAdam for the same-low-precision-dtype path.
//
// Companion to musa_applyadam_mixed_kernel.mu. The mixed variant assumes
// var/m/v are fp32 and only grad is low-precision; this kernel handles the
// case where ALL four (var, m, v, grad) are bf16 or fp16, which is what the
// stock ResourceApplyAdam op exposes when the user keeps optimizer state at
// the same dtype as the variables (e.g. tf.Variable(..., dtype=bfloat16)
// without a Keras mixed_bfloat16 policy).
//
// The kernel loads every value to fp32 in registers, performs the entire
// Adam update in fp32, and rounds the m / v / var stores back with RNE
// (__float2bfloat16 / __float2half from the MUSA SDK). This eliminates two
// classes of error from the legacy muDNN-Binary chain:
//   1) Per-op bf16 rounding on m, v, m_new, grad*grad, sqrt(v), update, ...
//      (the chain rounds back to bf16 nine times per step).
//   2) Quantization of (1 - beta1) to bf16 precision (only ~7 mantissa
//      bits), which is meaningful when beta1 = 0.999 because the result
//      (0.001) loses ~3 effective decimal digits.
//
// Numerics are bit-identical to running stock Adam in pure fp32 against a
// fp32 cast of the input bf16/fp16 tensors, except that the m/v/var stores
// round back to the source dtype at the end.

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

// Widening load: T -> float in register.
__device__ __forceinline__ float LoadF32(const float* p) { return *p; }

__device__ __forceinline__ float LoadF32(const Eigen::half* p) {
  const __half h = *reinterpret_cast<const __half*>(p);
  return __half2float(h);
}

__device__ __forceinline__ float LoadF32(const bfloat16* p) {
  const __mt_bfloat16 b = *reinterpret_cast<const __mt_bfloat16*>(p);
  return __bfloat162float(b);
}

// Narrowing store with RNE rounding (NOT bit-shift truncation, which was
// the bug pattern in the legacy custom kernels).
__device__ __forceinline__ void StoreFromF32(float* p, float v) { *p = v; }

__device__ __forceinline__ void StoreFromF32(Eigen::half* p, float v) {
  const __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ void StoreFromF32(bfloat16* p, float v) {
  const __mt_bfloat16 b = __float2bfloat16(v);
  *reinterpret_cast<__mt_bfloat16*>(p) = b;
}

template <typename T>
__global__ __launch_bounds__(256) void ApplyAdamSameTypeKernel(
    T* __restrict__ var, T* __restrict__ m, T* __restrict__ v,
    const T* __restrict__ grad, float lr_t, float beta1, float one_minus_b1,
    float beta2, float one_minus_b2, float epsilon, int64_t n,
    bool use_nesterov) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  const float g = LoadF32(grad + idx);
  const float m_val = LoadF32(m + idx);
  const float v_val = LoadF32(v + idx);
  const float var_val = LoadF32(var + idx);

  // m_t = beta1 * m + (1 - beta1) * g
  const float m_new = beta1 * m_val + one_minus_b1 * g;
  // v_t = beta2 * v + (1 - beta2) * g^2
  const float v_new = beta2 * v_val + one_minus_b2 * g * g;

  // Honour use_nesterov here for symmetry with MusaResourceApplyAdamMixed.
  // The legacy muDNN-Binary chain silently ignored it; this kernel does the
  // right thing if the attr is set. Callers that want bit-for-bit parity
  // with the old behavior should keep use_nesterov=false.
  const float numer = use_nesterov ? (beta1 * m_new + one_minus_b1 * g) : m_new;
  const float var_new = var_val - lr_t * numer / (sqrtf(v_new) + epsilon);

  StoreFromF32(m + idx, m_new);
  StoreFromF32(v + idx, v_new);
  StoreFromF32(var + idx, var_new);
}

template <typename T>
void LaunchApplyAdamSameTypeImpl(T* var, T* m, T* v, const T* grad, float lr_t,
                                  float beta1, float beta2, float epsilon,
                                  int64_t n, bool use_nesterov,
                                  musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  // Precompute (1 - beta) on the host so the kernel inner loop stays at the
  // minimum number of FMA-shaped ops.
  const float one_minus_b1 = 1.0f - beta1;
  const float one_minus_b2 = 1.0f - beta2;
  ApplyAdamSameTypeKernel<T><<<blocks, kThreads, 0, stream>>>(
      var, m, v, grad, lr_t, beta1, one_minus_b1, beta2, one_minus_b2, epsilon,
      n, use_nesterov);
}

}  // namespace

extern "C" {

void LaunchApplyAdamSameType_BFloat16(void* var, void* m, void* v,
                                       const void* grad, float lr_t,
                                       float beta1, float beta2, float epsilon,
                                       int64_t n, bool use_nesterov,
                                       musaStream_t stream) {
  LaunchApplyAdamSameTypeImpl<bfloat16>(
      reinterpret_cast<bfloat16*>(var), reinterpret_cast<bfloat16*>(m),
      reinterpret_cast<bfloat16*>(v),
      reinterpret_cast<const bfloat16*>(grad), lr_t, beta1, beta2, epsilon, n,
      use_nesterov, stream);
}

void LaunchApplyAdamSameType_Half(void* var, void* m, void* v,
                                   const void* grad, float lr_t, float beta1,
                                   float beta2, float epsilon, int64_t n,
                                   bool use_nesterov, musaStream_t stream) {
  LaunchApplyAdamSameTypeImpl<Eigen::half>(
      reinterpret_cast<Eigen::half*>(var), reinterpret_cast<Eigen::half*>(m),
      reinterpret_cast<Eigen::half*>(v),
      reinterpret_cast<const Eigen::half*>(grad), lr_t, beta1, beta2, epsilon,
      n, use_nesterov, stream);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
