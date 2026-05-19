// MUSA AddN Custom Kernel
// Performs element-wise addition of N tensors in a single kernel launch
// 
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

#define MAX_INLINE_ADDN_INPUTS 8

struct InlinePointers {
  const void* ptrs[MAX_INLINE_ADDN_INPUTS];
};

template <typename T>
__global__ void AddNKernelInline(InlinePointers inputs, T* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const T* p0 = static_cast<const T*>(inputs.ptrs[0]);
    T sum = p0[idx];
    switch (num_inputs) {
      case 8:
        sum += static_cast<const T*>(inputs.ptrs[7])[idx];
      case 7:
        sum += static_cast<const T*>(inputs.ptrs[6])[idx];
      case 6:
        sum += static_cast<const T*>(inputs.ptrs[5])[idx];
      case 5:
        sum += static_cast<const T*>(inputs.ptrs[4])[idx];
      case 4:
        sum += static_cast<const T*>(inputs.ptrs[3])[idx];
      case 3:
        sum += static_cast<const T*>(inputs.ptrs[2])[idx];
      case 2:
        sum += static_cast<const T*>(inputs.ptrs[1])[idx];
        break;
      default:
        for (int i = 1; i < num_inputs; ++i) {
          sum += static_cast<const T*>(inputs.ptrs[i])[idx];
        }
        break;
    }
    output[idx] = sum;
  }
}

__global__ void AddNKernelInlineFloatVec4(InlinePointers inputs, float* output,
                                          int num_inputs, int vec_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float4* p0 = reinterpret_cast<const float4*>(inputs.ptrs[0]);
    float4 sum = p0[idx];
    for (int i = 1; i < num_inputs; ++i) {
      const float4 v =
          reinterpret_cast<const float4*>(inputs.ptrs[i])[idx];
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    reinterpret_cast<float4*>(output)[idx] = sum;
  }
}


__global__ void AddNKernelInlineFloat8Vec4(InlinePointers inputs, float* output,
                                           int vec_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float4* p0 = reinterpret_cast<const float4*>(inputs.ptrs[0]);
    const float4* p1 = reinterpret_cast<const float4*>(inputs.ptrs[1]);
    const float4* p2 = reinterpret_cast<const float4*>(inputs.ptrs[2]);
    const float4* p3 = reinterpret_cast<const float4*>(inputs.ptrs[3]);
    const float4* p4 = reinterpret_cast<const float4*>(inputs.ptrs[4]);
    const float4* p5 = reinterpret_cast<const float4*>(inputs.ptrs[5]);
    const float4* p6 = reinterpret_cast<const float4*>(inputs.ptrs[6]);
    const float4* p7 = reinterpret_cast<const float4*>(inputs.ptrs[7]);

    float4 a0 = p0[idx];
    float4 a1 = p1[idx];
    float4 a2 = p2[idx];
    float4 a3 = p3[idx];
    float4 a4 = p4[idx];
    float4 a5 = p5[idx];
    float4 a6 = p6[idx];
    float4 a7 = p7[idx];

    float4 s;
    s.x = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x + a6.x + a7.x;
    s.y = a0.y + a1.y + a2.y + a3.y + a4.y + a5.y + a6.y + a7.y;
    s.z = a0.z + a1.z + a2.z + a3.z + a4.z + a5.z + a6.z + a7.z;
    s.w = a0.w + a1.w + a2.w + a3.w + a4.w + a5.w + a6.w + a7.w;

    reinterpret_cast<float4*>(output)[idx] = s;
  }
}

__global__ void AddNKernelInlineBFloat16(InlinePointers inputs, __mt_bfloat16* output, int num_inputs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sum = __bfloat162float(static_cast<const __mt_bfloat16*>(inputs.ptrs[0])[idx]);
    for (int i = 1; i < num_inputs; ++i) {
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
static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

void LaunchAddNKernelFloat(const float** inputs, InlinePointers inline_inputs,
                           float* output, int num_inputs, int size,
                           musaStream_t stream) {
  const int threads_per_block = 256;

  // Fast path for common AddN case in transformer workloads:
  // float, inline 8-input add, 16-byte aligned pointers.
  if (inputs == nullptr && num_inputs == 8 && size >= 4 && IsAligned16(output)) {
    bool all_aligned = true;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      all_aligned &= IsAligned16(inline_inputs.ptrs[i]);
    }
    if (all_aligned) {
      int vec_size = size / 4;
      int vec_blocks = (vec_size + threads_per_block - 1) / threads_per_block;
      AddNKernelInlineFloat8Vec4<<<vec_blocks, threads_per_block, 0, stream>>>(
          inline_inputs, output, vec_size);

      int tail = size - vec_size * 4;
      if (tail > 0) {
        InlinePointers tail_inputs = inline_inputs;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          tail_inputs.ptrs[i] =
              static_cast<const float*>(inline_inputs.ptrs[i]) + vec_size * 4;
        }
        int tail_blocks = (tail + threads_per_block - 1) / threads_per_block;
        AddNKernelInline<float><<<tail_blocks, threads_per_block, 0, stream>>>(
            tail_inputs, output + vec_size * 4, num_inputs, tail);
      }
      return;
    }
  }

  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  if (inputs == nullptr && num_inputs <= MAX_INLINE_ADDN_INPUTS) {
    if ((size % 4) == 0 && IsAligned16(output)) {
      bool all_aligned = true;
      for (int i = 0; i < num_inputs; ++i) {
        all_aligned &= IsAligned16(inline_inputs.ptrs[i]);
      }
      if (all_aligned) {
        const int vec_size = size / 4;
        const int vec_blocks =
            (vec_size + threads_per_block - 1) / threads_per_block;
        AddNKernelInlineFloatVec4<<<vec_blocks, threads_per_block, 0, stream>>>(
            inline_inputs, output, num_inputs, vec_size);
        return;
      }
    }
    AddNKernelInline<float><<<blocks, threads_per_block, 0, stream>>>(
        inline_inputs, output, num_inputs, size);
  } else {
    AddNKernelFloat<<<blocks, threads_per_block, 0, stream>>>(
        inputs, output, num_inputs, size);
  }
}


__global__ void AddNWithSliceGradKernelFloat4(
    InlinePointers base_inputs, const float* __restrict__ slice_grad,
    float* __restrict__ output, int num_base_inputs, int vec_size,
    int axis_dim, int slice_start, int inner_dim) {
  const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= vec_size) return;

  const int elem_idx = vec_idx * 4;
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int i = 0; i < num_base_inputs; ++i) {
    const float4 v =
        reinterpret_cast<const float4*>(base_inputs.ptrs[i])[vec_idx];
    sum.x += v.x;
    sum.y += v.y;
    sum.z += v.z;
    sum.w += v.w;
  }

  const int inner_idx = elem_idx % inner_dim;
  const int axis_idx = (elem_idx / inner_dim) % axis_dim;
  if (axis_idx >= slice_start) {
    const int outer_idx = elem_idx / (axis_dim * inner_dim);
    const int slice_axis_idx = axis_idx - slice_start;
    const int slice_axis_dim = axis_dim - slice_start;
    const int slice_elem_idx =
        (outer_idx * slice_axis_dim + slice_axis_idx) * inner_dim + inner_idx;
    const float4 sg =
        reinterpret_cast<const float4*>(slice_grad)[slice_elem_idx / 4];
    sum.x += sg.x;
    sum.y += sg.y;
    sum.z += sg.z;
    sum.w += sg.w;
  }

  reinterpret_cast<float4*>(output)[vec_idx] = sum;
}


__global__ void AddNWithSliceGradKernelFloat(
    InlinePointers base_inputs, const float* __restrict__ slice_grad,
    float* __restrict__ output, int num_base_inputs, int size, int axis_dim,
    int slice_start, int inner_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  float sum = 0.0f;
  for (int i = 0; i < num_base_inputs; ++i) {
    sum += static_cast<const float*>(base_inputs.ptrs[i])[idx];
  }

  const int inner_idx = idx % inner_dim;
  const int axis_idx = (idx / inner_dim) % axis_dim;
  if (axis_idx >= slice_start) {
    const int outer_idx = idx / (axis_dim * inner_dim);
    const int slice_axis_idx = axis_idx - slice_start;
    const int slice_axis_dim = axis_dim - slice_start;
    const int slice_idx =
        (outer_idx * slice_axis_dim + slice_axis_idx) * inner_dim + inner_idx;
    sum += slice_grad[slice_idx];
  }

  output[idx] = sum;
}

void LaunchAddNWithSliceGradFloat(InlinePointers base_inputs,
                                  const float* slice_grad, float* output,
                                  int num_base_inputs, int size,
                                  int axis_dim, int slice_start,
                                  int inner_dim, musaStream_t stream) {
  if (size <= 0) return;
  const int threads_per_block = 256;
  if ((size % 4) == 0 && (inner_dim % 4) == 0 && IsAligned16(output) &&
      IsAligned16(slice_grad)) {
    bool all_aligned = true;
    for (int i = 0; i < num_base_inputs; ++i) {
      all_aligned &= IsAligned16(base_inputs.ptrs[i]);
    }
    if (all_aligned) {
      const int vec_size = size / 4;
      const int vec_blocks =
          (vec_size + threads_per_block - 1) / threads_per_block;
      AddNWithSliceGradKernelFloat4<<<vec_blocks, threads_per_block, 0,
                                      stream>>>(
          base_inputs, slice_grad, output, num_base_inputs, vec_size, axis_dim,
          slice_start, inner_dim);
      return;
    }
  }

  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  AddNWithSliceGradKernelFloat<<<blocks, threads_per_block, 0, stream>>>(
      base_inputs, slice_grad, output, num_base_inputs, size, axis_dim,
      slice_start, inner_dim);
}

__global__ void AddNWithSliceGradKernelBFloat16(
    InlinePointers base_inputs, const __mt_bfloat16* __restrict__ slice_grad,
    __mt_bfloat16* __restrict__ output, int num_base_inputs, int size,
    int axis_dim, int slice_start, int inner_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  float sum = 0.0f;
  for (int i = 0; i < num_base_inputs; ++i) {
    sum += __bfloat162float(
        static_cast<const __mt_bfloat16*>(base_inputs.ptrs[i])[idx]);
  }

  const int inner_idx = idx % inner_dim;
  const int axis_idx = (idx / inner_dim) % axis_dim;
  if (axis_idx >= slice_start) {
    const int outer_idx = idx / (axis_dim * inner_dim);
    const int slice_axis_idx = axis_idx - slice_start;
    const int slice_axis_dim = axis_dim - slice_start;
    const int slice_idx =
        (outer_idx * slice_axis_dim + slice_axis_idx) * inner_dim + inner_idx;
    sum += __bfloat162float(slice_grad[slice_idx]);
  }

  output[idx] = __float2bfloat16(sum);
}

void LaunchAddNWithSliceGradBFloat16(InlinePointers base_inputs,
                                     const void* slice_grad, void* output,
                                     int num_base_inputs, int size,
                                     int axis_dim, int slice_start,
                                     int inner_dim, musaStream_t stream) {
  if (size <= 0) return;
  const int threads_per_block = 256;
  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  AddNWithSliceGradKernelBFloat16<<<blocks, threads_per_block, 0, stream>>>(
      base_inputs, reinterpret_cast<const __mt_bfloat16*>(slice_grad),
      reinterpret_cast<__mt_bfloat16*>(output), num_base_inputs, size,
      axis_dim, slice_start, inner_dim);
}

__global__ void ConcatWithSliceGradKernelFloat(
    const float* __restrict__ slice_grad, const float* __restrict__ input1,
    const float* __restrict__ input2, float* __restrict__ output, int outer,
    int axis_dim, int slice_start, int inner_dim) {
  const int size = outer * axis_dim * inner_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const int inner_idx = idx % inner_dim;
  const int axis_idx = (idx / inner_dim) % axis_dim;
  const int outer_idx = idx / (axis_dim * inner_dim);
  const int out_idx = (outer_idx * axis_dim + axis_idx) * inner_dim * 3 +
                      inner_idx;

  if (axis_idx < slice_start) {
    output[out_idx] = 0.0f;
  } else {
    const int slice_axis_dim = axis_dim - slice_start;
    const int slice_idx =
        (outer_idx * slice_axis_dim + (axis_idx - slice_start)) *
            inner_dim +
        inner_idx;
    output[out_idx] = slice_grad[slice_idx];
  }
  output[out_idx + inner_dim] = input1[idx];
  output[out_idx + inner_dim * 2] = input2[idx];
}

void LaunchConcatWithSliceGradFloat(const float* slice_grad,
                                    const float* input1, const float* input2,
                                    float* output, int outer, int axis_dim,
                                    int slice_start, int inner_dim,
                                    musaStream_t stream) {
  const int size = outer * axis_dim * inner_dim;
  if (size <= 0) return;
  const int threads_per_block = 256;
  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  ConcatWithSliceGradKernelFloat<<<blocks, threads_per_block, 0, stream>>>(
      slice_grad, input1, input2, output, outer, axis_dim, slice_start,
      inner_dim);
}

__global__ void ConcatWithSliceGradKernelBFloat16(
    const __mt_bfloat16* __restrict__ slice_grad,
    const __mt_bfloat16* __restrict__ input1,
    const __mt_bfloat16* __restrict__ input2,
    __mt_bfloat16* __restrict__ output, int outer, int axis_dim,
    int slice_start, int inner_dim) {
  const int size = outer * axis_dim * inner_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const int inner_idx = idx % inner_dim;
  const int axis_idx = (idx / inner_dim) % axis_dim;
  const int outer_idx = idx / (axis_dim * inner_dim);
  const int out_idx = (outer_idx * axis_dim + axis_idx) * inner_dim * 3 +
                      inner_idx;

  if (axis_idx < slice_start) {
    output[out_idx] = __float2bfloat16(0.0f);
  } else {
    const int slice_axis_dim = axis_dim - slice_start;
    const int slice_idx =
        (outer_idx * slice_axis_dim + (axis_idx - slice_start)) *
            inner_dim +
        inner_idx;
    output[out_idx] = slice_grad[slice_idx];
  }
  output[out_idx + inner_dim] = input1[idx];
  output[out_idx + inner_dim * 2] = input2[idx];
}

void LaunchConcatWithSliceGradBFloat16(
    const void* slice_grad, const void* input1, const void* input2,
    void* output, int outer, int axis_dim, int slice_start, int inner_dim,
    musaStream_t stream) {
  const int size = outer * axis_dim * inner_dim;
  if (size <= 0) return;
  const int threads_per_block = 256;
  const int blocks = (size + threads_per_block - 1) / threads_per_block;
  ConcatWithSliceGradKernelBFloat16<<<blocks, threads_per_block, 0, stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(slice_grad),
      reinterpret_cast<const __mt_bfloat16*>(input1),
      reinterpret_cast<const __mt_bfloat16*>(input2),
      reinterpret_cast<__mt_bfloat16*>(output), outer, axis_dim, slice_start,
      inner_dim);
}


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

DEFINE_ADDN_LAUNCHER(LaunchAddNKernelDouble, AddNKernelDouble, AddNKernelInline<double>, double)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelHalf, AddNKernelHalf, AddNKernelInline<half>, half)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelBFloat16, AddNKernelBFloat16, AddNKernelInlineBFloat16, __mt_bfloat16)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt32, AddNKernelInt32, AddNKernelInline<int>, int)
DEFINE_ADDN_LAUNCHER(LaunchAddNKernelInt64, AddNKernelInt64, AddNKernelInline<int64_t>, int64_t)

#undef DEFINE_ADDN_LAUNCHER

}  // extern "C"
