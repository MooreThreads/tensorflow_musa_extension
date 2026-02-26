// MUSA Range Custom Kernel
// Generates a sequence of values on device (no host computation)
// 
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>

extern "C" {

// Float kernel
__global__ void RangeKernelFloat(float* output, float start, float delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = start + static_cast<float>(idx) * delta;
  }
}

// Double kernel
__global__ void RangeKernelDouble(double* output, double start, double delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = start + static_cast<double>(idx) * delta;
  }
}

// Int32 kernel
__global__ void RangeKernelInt32(int* output, int start, int delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = start + idx * delta;
  }
}

// Int64 kernel
__global__ void RangeKernelInt64(long long* output, long long start, long long delta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = start + static_cast<long long>(idx) * delta;
  }
}

// Launcher functions
#define DEFINE_RANGE_LAUNCHER(name, kernel, T) \
  void name(T* output, T start, T delta, int size, musaStream_t stream) { \
    const int threads_per_block = 256; \
    const int blocks = (size + threads_per_block - 1) / threads_per_block; \
    kernel<<<blocks, threads_per_block, 0, stream>>>(output, start, delta, size); \
  }

DEFINE_RANGE_LAUNCHER(LaunchRangeKernelFloat, RangeKernelFloat, float)
DEFINE_RANGE_LAUNCHER(LaunchRangeKernelDouble, RangeKernelDouble, double)
DEFINE_RANGE_LAUNCHER(LaunchRangeKernelInt32, RangeKernelInt32, int)
DEFINE_RANGE_LAUNCHER(LaunchRangeKernelInt64, RangeKernelInt64, long long)

#undef DEFINE_RANGE_LAUNCHER

}  // extern "C"
