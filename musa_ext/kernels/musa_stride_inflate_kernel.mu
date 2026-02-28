#include <musa_runtime.h>

#include "tensorflow/core/framework/bfloat16.h"

#include "musa_stride_inflate_kernel.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename T>
__global__ void MusaStrideKernel(int64_t size, const T* __restrict__ in,
                                 T* __restrict__ out, DimSizeArray dims,
                                 DimSizeArray strides, int ndims) {
  const int64_t block_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < size) {
    int64_t coords[kMaxStrideInflateDims] = {0};
    int64_t tmp = idx;
    for (int dim = ndims - 1; dim >= 0; --dim) {
      const int64_t dim_size = dims.value[dim];
      if (dim_size > 0) {
        coords[dim] = tmp % dim_size;
        tmp /= dim_size;
      }
    }
    int64_t in_index = 0;
    for (int dim = 0; dim < ndims; ++dim) {
      in_index += coords[dim] * strides.value[dim];
    }
    out[idx] = in[in_index];
    idx += block_stride;
  }
}

template <typename T>
__global__ void MusaInflateKernel(int64_t size, const T* __restrict__ in,
                                  T* __restrict__ out, DimSizeArray in_dims,
                                  DimSizeArray strides, int ndims,
                                  int64_t out_size) {
  const int64_t block_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < size) {
    int64_t coords[kMaxStrideInflateDims] = {0};
    int64_t tmp = idx;
    for (int dim = ndims - 1; dim >= 0; --dim) {
      const int64_t dim_size = in_dims.value[dim];
      if (dim_size > 0) {
        coords[dim] = tmp % dim_size;
        tmp /= dim_size;
      }
    }
    int64_t out_index = 0;
    for (int dim = 0; dim < ndims; ++dim) {
      out_index += coords[dim] * strides.value[dim];
    }
    if (out_index >= 0 && out_index < out_size) {
      out[out_index] = in[idx];
    }
    idx += block_stride;
  }
}

}  // namespace

template <typename T>
void MusaStrideKernelLauncher(musaStream_t stream, int64_t size, const T* in,
                              T* out, DimSizeArray dims,
                              DimSizeArray strides, int ndims) {
  if (size == 0) {
    return;
  }
  constexpr int block_size = 256;
  const int grid_size = static_cast<int>(
      (size + block_size - 1) / block_size);
  MusaStrideKernel<T><<<grid_size, block_size, 0, stream>>>(
      size, in, out, dims, strides, ndims);
}

template <typename T>
void MusaInflateKernelLauncher(musaStream_t stream, int64_t input_size,
                               const T* in, T* out, DimSizeArray in_dims,
                               DimSizeArray strides, int ndims,
                               int64_t out_size) {
  if (input_size == 0) {
    return;
  }
  constexpr int block_size = 256;
  const int grid_size = static_cast<int>(
      (input_size + block_size - 1) / block_size);
  MusaInflateKernel<T><<<grid_size, block_size, 0, stream>>>(
      input_size, in, out, in_dims, strides, ndims, out_size);
}

#define INSTANTIATE_STRIDE_INFLATE(T)                                       \
  template void MusaStrideKernelLauncher<T>(musaStream_t, int64_t, const T*,  \
                                            T*, DimSizeArray, DimSizeArray,   \
                                            int);                             \
  template void MusaInflateKernelLauncher<T>(                                \
      musaStream_t, int64_t, const T*, T*, DimSizeArray, DimSizeArray, int,    \
      int64_t)

INSTANTIATE_STRIDE_INFLATE(float);
INSTANTIATE_STRIDE_INFLATE(double);
INSTANTIATE_STRIDE_INFLATE(int32);
INSTANTIATE_STRIDE_INFLATE(int64);
INSTANTIATE_STRIDE_INFLATE(Eigen::half);
INSTANTIATE_STRIDE_INFLATE(Eigen::bfloat16);

#undef INSTANTIATE_STRIDE_INFLATE

}  // namespace musa
}  // namespace tensorflow
