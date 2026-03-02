// MUSA AddN Custom Kernel
// Performs element-wise addition of N tensors in a single kernel launch
// 
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

extern "C" {

// Float kernel
__global__ void AddNKernelFloat(const float** inputs, float* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sum = inputs[0][idx];
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Double kernel
__global__ void AddNKernelDouble(const double** inputs, double* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double sum = inputs[0][idx];
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Half (float16) kernel
__global__ void AddNKernelHalf(const half** inputs, half* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    half sum = inputs[0][idx];
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// BFloat16 kernel (needs float accumulation for precision)
__global__ void AddNKernelBFloat16(const __mt_bfloat16** inputs, 
                                    __mt_bfloat16* output, 
                                    int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sum = __bfloat162float(inputs[0][idx]);
    for (int i = 1; i < num_inputs; ++i) {
      sum += __bfloat162float(inputs[i][idx]);
    }
    output[idx] = __float2bfloat16(sum);
  }
}

// Int32 kernel
__global__ void AddNKernelInt32(const int** inputs, int* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int sum = inputs[0][idx];
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Int64 kernel
__global__ void AddNKernelInt64(const long long** inputs, long long* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    long long sum = inputs[0][idx];
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Launcher functions - called from C++ code
#define DEFINE_ADDN_LAUNCHER(name, kernel, T) \
  void name(const T** inputs, T* output, int num_inputs, int size, musaStream_t stream) { \
    const int threads_per_block = 256; \
    const int blocks = (size + threads_per_block - 1) / threads_per_block; \
    kernel<<<blocks, threads_per_block, 0, stream>>>(inputs, output, num_inputs, size); \
  }

DEFINE_ADDN_LAUNCHER(LaunchAddNKernelFloat, AddNKernelFloat, float)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelDouble, AddNKernelDouble, double)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelHalf, AddNKernelHalf, half)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelBFloat16, AddNKernelBFloat16, __mt_bfloat16)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt32, AddNKernelInt32, int)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt64, AddNKernelInt64, long long)

#undef DEFINE_ADDN_LAUNCHER

}  // extern "C"
