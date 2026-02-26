// MUSA Softplus Custom Kernel
// Performs element-wise softplus: softplus(x) = log(1 + exp(x))
// Numerically stable form: max(x, 0) + log1p(exp(-abs(x)))
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <math.h>

extern "C" {

// ----------------------------------------------------------------------------
// Numerically stable softplus helpers
// ----------------------------------------------------------------------------

__device__ __forceinline__ float SoftplusStable(float x) {
  float ax = fabsf(x);
  float mx = x > 0.0f ? x : 0.0f;
  return mx + log1pf(expf(-ax));
}

__device__ __forceinline__ double SoftplusStable(double x) {
  double ax = fabs(x);
  double mx = x > 0.0 ? x : 0.0;
  return mx + log1p(exp(-ax));
}

// ----------------------------------------------------------------------------
// Float kernel
// ----------------------------------------------------------------------------

__global__ void SoftplusKernelFloat(const float* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = SoftplusStable(input[idx]);
  }
}

// ----------------------------------------------------------------------------
// Double kernel
// ----------------------------------------------------------------------------

__global__ void SoftplusKernelDouble(const double* input, double* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = SoftplusStable(input[idx]);
  }
}

// ----------------------------------------------------------------------------
// Half (float16) kernel - compute in float for stability/precision
// ----------------------------------------------------------------------------

__global__ void SoftplusKernelHalf(const half* input, half* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = __half2float(input[idx]);
    float y = SoftplusStable(x);
    output[idx] = __float2half(y);
  }
}

// ----------------------------------------------------------------------------
// BFloat16 kernel - compute in float for stability/precision
// ----------------------------------------------------------------------------

__global__ void SoftplusKernelBFloat16(const __mt_bfloat16* input,
                                       __mt_bfloat16* output,
                                       int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = __bfloat162float(input[idx]);
    float y = SoftplusStable(x);
    output[idx] = __float2bfloat16(y);
  }
}

// ----------------------------------------------------------------------------
// Launcher functions - called from C++ code
// ----------------------------------------------------------------------------

#define DEFINE_SOFTPLUS_LAUNCHER(name, kernel, T)                              \
  void name(const T* input, T* output, int size, musaStream_t stream) {        \
    const int threads_per_block = 256;                                          \
    const int blocks = (size + threads_per_block - 1) / threads_per_block;     \
    kernel<<<blocks, threads_per_block, 0, stream>>>(input, output, size);     \
  }

DEFINE_SOFTPLUS_LAUNCHER(LaunchSoftplusKernelFloat, SoftplusKernelFloat, float)
DEFINE_SOFTPLUS_LAUNCHER(LaunchSoftplusKernelDouble, SoftplusKernelDouble, double)
DEFINE_SOFTPLUS_LAUNCHER(LaunchSoftplusKernelHalf, SoftplusKernelHalf, half)
DEFINE_SOFTPLUS_LAUNCHER(LaunchSoftplusKernelBFloat16, SoftplusKernelBFloat16, __mt_bfloat16)

#undef DEFINE_SOFTPLUS_LAUNCHER

}  // extern "C"