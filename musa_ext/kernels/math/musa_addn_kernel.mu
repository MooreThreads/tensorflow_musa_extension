// MUSA AddN Custom Kernel
// Performs element-wise addition of N tensors in a single kernel launch
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

#define MAX_INLINE_ADDN_INPUTS 16

struct InlinePointers {
  const void* ptrs[MAX_INLINE_ADDN_INPUTS];
};

template <typename T>
__global__ void AddNKernelInline(InlinePointers inputs, T* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T sum = T(0);
    for (int i = 0; i < num_inputs; ++i) {
      sum += static_cast<const T*>(inputs.ptrs[i])[idx];
    }
    output[idx] = sum;
  }
}

__global__ void AddNKernelInlineBFloat16(InlinePointers inputs, __mt_bfloat16* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sum = 0.0f;
    for (int i = 0; i < num_inputs; ++i) {
      sum += __bfloat162float(static_cast<const __mt_bfloat16*>(inputs.ptrs[i])[idx]);
    }
    output[idx] = __float2bfloat16(sum);
  }
}

extern "C" {

// Float kernel
__global__ void AddNKernelFloat(const float** inputs, float* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sum = inputs[0][idx];
    #pragma unroll
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
    #pragma unroll
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
    #pragma unroll
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
    #pragma unroll
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
    #pragma unroll
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Int64 kernel
__global__ void AddNKernelInt64(const int64_t** inputs, int64_t* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int64_t sum = inputs[0][idx];
    #pragma unroll
    for (int i = 1; i < num_inputs; ++i) {
      sum += inputs[i][idx];
    }
    output[idx] = sum;
  }
}

// Launcher functions - called from C++ code
#define DEFINE_ADDN_LAUNCHER(name, kernel, inline_kernel, T) \
  void name(const T** inputs, InlinePointers inline_inputs, T* output, int num_inputs, int size, musaStream_t stream) { \
    const int threads_per_block = 256; \
    const int blocks = (size + threads_per_block - 1) / threads_per_block; \
    if (inputs == nullptr && num_inputs <= MAX_INLINE_ADDN_INPUTS) { \
      inline_kernel<<<blocks, threads_per_block, 0, stream>>>(inline_inputs, output, num_inputs, size); \
    } else { \
      kernel<<<blocks, threads_per_block, 0, stream>>>(inputs, output, num_inputs, size); \
    } \
  }

DEFINE_ADDN_LAUNCHER(LaunchAddNKernelFloat, AddNKernelFloat, AddNKernelInline<float>, float)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelDouble, AddNKernelDouble, AddNKernelInline<double>, double)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelHalf, AddNKernelHalf, AddNKernelInline<half>, half)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelBFloat16, AddNKernelBFloat16, AddNKernelInlineBFloat16, __mt_bfloat16)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt32, AddNKernelInt32, AddNKernelInline<int>, int)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt64, AddNKernelInt64, AddNKernelInline<int64_t>, int64_t)

#undef DEFINE_ADDN_LAUNCHER

}  // extern "C"
