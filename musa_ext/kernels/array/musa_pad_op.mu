// MUSA Pad Custom Kernel
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>

#ifndef MAX_DIM
#define MAX_DIM 4  // NHWC 4d support only
#endif

template <typename T>
__global__ void nd_pad_kernel(const T *__restrict__ in, T *__restrict__ out,
                              const int real_dim, const int64_t *in_dims,
                              const int64_t *out_dims,
                              const int64_t *pad_before,
                              const int64_t *pad_after, const T pad_val,
                              const int64_t total_out_elements) {
  int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= total_out_elements) {
    return;
  }

  int64_t in_stride[MAX_DIM] = {1, 1, 1, 1};
  int64_t out_stride[MAX_DIM] = {1, 1, 1, 1};
  for (int d = real_dim - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_dims[d + 1];
    out_stride[d] = out_stride[d + 1] * out_dims[d + 1];
  }

  int64_t remain = global_idx;
  int64_t in_idx = 0;
  bool is_inside = true;

  for (int d = 0; d < real_dim; ++d) {
    int64_t out_coord = remain / out_stride[d];
    remain = remain % out_stride[d];

    int64_t in_coord = out_coord - pad_before[d];
    if (in_coord < 0 || in_coord >= in_dims[d]) {
      is_inside = false;
      break;
    }

    in_idx += in_coord * in_stride[d];
  }

  if (is_inside) {
    out[global_idx] = in[in_idx];
  } else {
    out[global_idx] = pad_val;
  }
}

#define INSTANTIATE_ND_PAD_KERNEL(T) \
  template __global__ void nd_pad_kernel<T>( \
      const T *__restrict__ in, T *__restrict__ out, const int real_dim, \
      const int64_t *in_dims, const int64_t *out_dims, \
      const int64_t *pad_before, const int64_t *pad_after, const T pad_val, \
      const int64_t total_out_elements);

INSTANTIATE_ND_PAD_KERNEL(float)
INSTANTIATE_ND_PAD_KERNEL(double)
INSTANTIATE_ND_PAD_KERNEL(int32_t)
INSTANTIATE_ND_PAD_KERNEL(int64_t)
INSTANTIATE_ND_PAD_KERNEL(uint8_t)

#undef INSTANTIATE_ND_PAD_KERNEL

extern "C" {
#define DEFINE_ND_PAD_LAUNCHER(T, name_suffix) \
  void nd_pad_kernel_launcher_##name_suffix(const T *input_data, T *output_data, \
                                            const int dims, const int64_t *in_dims, \
                                            const int64_t *out_dims, const int64_t *pad_before, \
                                            const int64_t *pad_after, const T pad_value, \
                                            const int64_t total_out_elements, const musaStream_t stream) { \
    const int block_size = 256; \
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size; \
    nd_pad_kernel<T><<<grid_size, block_size, 0, stream>>>( \
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, \
        pad_value, total_out_elements); \
  }

DEFINE_ND_PAD_LAUNCHER(float, float)
DEFINE_ND_PAD_LAUNCHER(double, double)
DEFINE_ND_PAD_LAUNCHER(int32_t, int32)
DEFINE_ND_PAD_LAUNCHER(int64_t, int64)
DEFINE_ND_PAD_LAUNCHER(uint8_t, uint8)

#undef DEFINE_ND_PAD_LAUNCHER
} // extern "C"
