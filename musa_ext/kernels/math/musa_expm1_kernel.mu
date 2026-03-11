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

/**
 * @file musa_expm1_kernel.mu
 * @brief MUSA kernel for Expm1 operation.
 *
 * Computes exp(x) - 1 with improved numerical precision for small x.
 */

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <math.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

// Helper functions for type conversion

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
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
    uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
    *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
    return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
    uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
    uint16_t b_val = *f_ptr >> 16;
    *reinterpret_cast<uint16_t*>(p) = b_val;
}

/**
 * @brief Compute expm1(x) = exp(x) - 1 with improved precision for small x.
 *
 * For small |x| values, use Taylor expansion to avoid precision loss:
 * expm1(x) = x + x^2/2! + x^3/3! + ...
 *
 * For larger |x| values, use standard expf(x) - 1.
 */
__device__ __forceinline__ float DeviceExpm1(float x) {
    // Threshold for using Taylor expansion
    // For |x| < 0.01, Taylor series gives better precision
    constexpr float kThreshold = 0.01f;

    float abs_x = fabsf(x);
    if (abs_x < kThreshold) {
        // Use Taylor expansion for small x: expm1(x) ≈ x + x^2/2 + x^3/6 + x^4/24
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return x + x2 * 0.5f + x3 * (1.0f / 6.0f) + x4 * (1.0f / 24.0f);
    } else {
        return expf(x) - 1.0f;
    }
}

/**
 * @brief Expm1 kernel for floating point types (float, half, bfloat16).
 */
template <typename T>
__global__ void Expm1Kernel(const T* src, T* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = LoadFloat(&src[i]);
        float result = DeviceExpm1(val);
        StoreFloat(&dst[i], result);
    }
}

/**
 * @brief Expm1 kernel specialization for double precision.
 */
template <>
__global__ void Expm1Kernel<double>(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = exp(src[i]) - 1.0;
    }
}

/**
 * @brief Expm1 kernel for integer types.
 * Integer inputs are converted to float for computation.
 */
template <typename T>
__global__ void Expm1IntKernel(const T* src, T* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = static_cast<float>(src[i]);
        float result = expf(val) - 1.0f;
        dst[i] = static_cast<T>(roundf(result));
    }
}

/**
 * @brief Kernel launcher for floating point types.
 */
template <typename T>
void LaunchExpm1(const T* src, T* dst, int n, musaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    Expm1Kernel<T><<<blocks, threads, 0, stream>>>(src, dst, n);
}

/**
 * @brief Kernel launcher for integer types.
 */
template <typename T>
void LaunchExpm1Int(const T* src, T* dst, int n, musaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    Expm1IntKernel<T><<<blocks, threads, 0, stream>>>(src, dst, n);
}

// Explicit template instantiation for floating point types
template void LaunchExpm1<float>(const float*, float*, int, musaStream_t);
template void LaunchExpm1<double>(const double*, double*, int, musaStream_t);
template void LaunchExpm1<Eigen::half>(const Eigen::half*, Eigen::half*, int, musaStream_t);
template void LaunchExpm1<bfloat16>(const bfloat16*, bfloat16*, int, musaStream_t);

// Explicit template instantiation for integer types
template void LaunchExpm1Int<int32>(const int32*, int32*, int, musaStream_t);
template void LaunchExpm1Int<long>(const long*, long*, int, musaStream_t);
template void LaunchExpm1Int<long long>(const long long*, long long*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow