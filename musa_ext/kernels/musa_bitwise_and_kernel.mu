#include <musa_runtime.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

constexpr int kMaxBroadcastDims = 8;


template <typename T>
__global__ void MusaBitwiseAndKernel(const T* input_a,
                                     const T* input_b,
                                     T* output,
                                     int64_t size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input_a[idx] & input_b[idx];
  }
}

template <typename T>
void MusaBitwiseAndKernelLauncher(const T* input_a, const T* input_b,
                                  T* output, int64_t size) {
  if (size == 0) return;

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;

  MusaBitwiseAndKernel<T><<<num_blocks, block_size, 0>>>(input_a, input_b,
                                                          output, size);
}


__device__ __forceinline__
int64_t compute_input_index(int64_t out_idx,
                            const int64_t* output_shape,
                            const int64_t* strides,
                            int ndims) {
  int64_t in_idx = 0;
  int64_t remaining = out_idx;
  for (int d = 0; d < ndims; ++d) {
    int64_t dim_stride = 1;
    for (int k = d + 1; k < ndims; ++k) {
      dim_stride *= output_shape[k];
    }
    int64_t coord = remaining / dim_stride;
    remaining = remaining % dim_stride;
    in_idx += coord * strides[d];
  }
  return in_idx;
}

template <typename T>
__global__ void MusaBitwiseAndBroadcastKernel(
    const T* input_a,
    const T* input_b,
    T* output,
    const int64_t* output_shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndims,
    int64_t size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int64_t a_idx = compute_input_index(idx, output_shape, a_strides, ndims);
    int64_t b_idx = compute_input_index(idx, output_shape, b_strides, ndims);
    output[idx] = input_a[a_idx] & input_b[b_idx];
  }
}

template <typename T>
void MusaBitwiseAndBroadcastKernelLauncher(
    const T* input_a, const T* input_b, T* output,
    const int64_t* output_shape, const int64_t* a_strides,
    const int64_t* b_strides, int ndims, int64_t size) {
  if (size == 0) return;

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;

  MusaBitwiseAndBroadcastKernel<T><<<num_blocks, block_size, 0>>>(
      input_a, input_b, output,
      output_shape, a_strides, b_strides,
      ndims, size);
}


template void MusaBitwiseAndKernelLauncher<int8>(const int8*, const int8*,
    int8*, int64_t);
template void MusaBitwiseAndKernelLauncher<int16>(const int16*, const int16*,
    int16*, int64_t);
template void MusaBitwiseAndKernelLauncher<int32>(const int32*, const int32*,
    int32*, int64_t);
template void MusaBitwiseAndKernelLauncher<long long>(const long long*, const long long*,
    long long*, int64_t);
template void MusaBitwiseAndKernelLauncher<uint8>(const uint8*, const uint8*,
    uint8*, int64_t);
template void MusaBitwiseAndKernelLauncher<uint16>(const uint16*, const uint16*,
    uint16*, int64_t);
template void MusaBitwiseAndKernelLauncher<uint32>(const uint32*, const uint32*,
    uint32*, int64_t);
template void MusaBitwiseAndKernelLauncher<unsigned long long>(const unsigned long long*, const unsigned long long*,
    unsigned long long*, int64_t);


template void MusaBitwiseAndBroadcastKernelLauncher<int8>(
    const int8*, const int8*, int8*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<int16>(
    const int16*, const int16*, int16*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<int32>(
    const int32*, const int32*, int32*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<long long>(
    const long long*, const long long*, long long*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<uint8>(
    const uint8*, const uint8*, uint8*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<uint16>(
    const uint16*, const uint16*, uint16*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<uint32>(
    const uint32*, const uint32*, uint32*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);
template void MusaBitwiseAndBroadcastKernelLauncher<unsigned long long>(
    const unsigned long long*, const unsigned long long*, unsigned long long*,
    const int64_t*, const int64_t*, const int64_t*, int, int64_t);

}  // namespace musa
}  // namespace tensorflow