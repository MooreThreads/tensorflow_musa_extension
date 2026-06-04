#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

constexpr int kMaxInlineTileDims = 8;

struct TileDims {
  int64_t input[kMaxInlineTileDims];
  int64_t output[kMaxInlineTileDims];
};

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename T>
__global__ void TileKernel(const T* __restrict__ input,
                           const int64_t* __restrict__ input_dims,
                           const int64_t* __restrict__ output_dims, int dims,
                           int64_t output_size, T* __restrict__ output) {
  const int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (tid >= output_size) return;

  int64_t remaining = tid;
  int64_t input_offset = 0;
  int64_t input_stride = 1;

  for (int d = dims - 1; d >= 0; --d) {
    const int64_t out_dim = output_dims[d];
    const int64_t in_dim = input_dims[d];
    const int64_t out_coord = remaining % out_dim;
    remaining /= out_dim;
    input_offset += (out_coord % in_dim) * input_stride;
    input_stride *= in_dim;
  }

  output[tid] = input[input_offset];
}

template <typename T>
__global__ void TileSmallDimsKernel(const T* __restrict__ input,
                                    TileDims tile_dims, int dims,
                                    int64_t output_size,
                                    T* __restrict__ output) {
  const int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (tid >= output_size) return;

  if (dims == 3) {
    const int64_t in0 = tile_dims.input[0];
    const int64_t in1 = tile_dims.input[1];
    const int64_t in2 = tile_dims.input[2];
    const int64_t out0 = tile_dims.output[0];
    const int64_t out1 = tile_dims.output[1];
    const int64_t out2 = tile_dims.output[2];

    const int64_t c = tid % out2;
    const int64_t q = tid / out2;
    const int64_t b = q % out1;
    const int64_t a = q / out1;
    const int64_t i0 = (in0 == out0) ? a : (in0 == 1 ? 0 : a % in0);
    const int64_t i1 = (in1 == out1) ? b : (in1 == 1 ? 0 : b % in1);
    const int64_t i2 = (in2 == out2) ? c : (in2 == 1 ? 0 : c % in2);
    output[tid] = input[(i0 * in1 + i1) * in2 + i2];
    return;
  }

  if (dims == 2) {
    const int64_t in0 = tile_dims.input[0];
    const int64_t in1 = tile_dims.input[1];
    const int64_t out0 = tile_dims.output[0];
    const int64_t out1 = tile_dims.output[1];

    const int64_t b = tid % out1;
    const int64_t a = tid / out1;
    const int64_t i0 = (in0 == out0) ? a : (in0 == 1 ? 0 : a % in0);
    const int64_t i1 = (in1 == out1) ? b : (in1 == 1 ? 0 : b % in1);
    output[tid] = input[i0 * in1 + i1];
    return;
  }

  int64_t remaining = tid;
  int64_t input_offset = 0;
  int64_t input_stride = 1;

  for (int d = dims - 1; d >= 0; --d) {
    const int64_t out_dim = tile_dims.output[d];
    const int64_t in_dim = tile_dims.input[d];
    const int64_t out_coord = remaining % out_dim;
    remaining /= out_dim;
    input_offset += (out_coord % in_dim) * input_stride;
    input_stride *= in_dim;
  }

  output[tid] = input[input_offset];
}

template <typename T>
__global__ void TileLastDimKernel(const T* __restrict__ input, int64_t rows,
                                  int64_t width, int64_t output_size,
                                  T* __restrict__ output) {
  int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (; tid < output_size; tid += stride) {
    output[tid] = input[tid / width];
  }
}

template <typename T>
__global__ void TileLastDim128Kernel(const T* __restrict__ input, int64_t rows,
                                     T* __restrict__ output) {
  int64_t tid =
      blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t output_size = rows * 128;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (; tid < output_size; tid += stride) {
    output[tid] = input[tid >> 7];
  }
}

}  // namespace

template <typename T>
void LaunchMusaTileSmallDimsKernel(const T* input, TileDims tile_dims, int dims,
                                   int64_t output_size, T* output,
                                   musaStream_t stream) {
  if (output_size <= 0) return;
  const int64_t blocks =
      (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  TileSmallDimsKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, tile_dims, dims, output_size, output);
}

template <typename T>
void LaunchMusaTileLastDimKernel(const T* input, int64_t rows, int64_t width,
                                 int64_t output_size, T* output,
                                 musaStream_t stream) {
  if (output_size <= 0 || rows <= 0 || width <= 1) return;
  const int64_t blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (width == 128 && output_size == rows * 128) {
    TileLastDim128Kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
        input, rows, output);
    return;
  }
  TileLastDimKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, rows, width, output_size, output);
}

template <typename T>
void LaunchMusaTileKernel(const T* input, const int64_t* input_dims,
                          const int64_t* output_dims, int dims,
                          int64_t output_size, T* output,
                          musaStream_t stream) {
  if (output_size <= 0) return;
  const int64_t blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  TileKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, input_dims, output_dims, dims, output_size, output);
}

template void LaunchMusaTileKernel<float>(const float*, const int64_t*,
                                          const int64_t*, int, int64_t, float*,
                                          musaStream_t);
template void LaunchMusaTileKernel<Eigen::half>(
    const Eigen::half*, const int64_t*, const int64_t*, int, int64_t,
    Eigen::half*, musaStream_t);
template void LaunchMusaTileKernel<bfloat16>(
    const bfloat16*, const int64_t*, const int64_t*, int, int64_t, bfloat16*,
    musaStream_t);
template void LaunchMusaTileKernel<double>(const double*, const int64_t*,
                                           const int64_t*, int, int64_t,
                                           double*, musaStream_t);
template void LaunchMusaTileKernel<int32>(const int32*, const int64_t*,
                                          const int64_t*, int, int64_t, int32*,
                                          musaStream_t);
template void LaunchMusaTileKernel<int64>(const int64*, const int64_t*,
                                          const int64_t*, int, int64_t, int64*,
                                          musaStream_t);
template void LaunchMusaTileKernel<bool>(const bool*, const int64_t*,
                                         const int64_t*, int, int64_t, bool*,
                                         musaStream_t);

template void LaunchMusaTileSmallDimsKernel<float>(const float*, TileDims, int,
                                                   int64_t, float*,
                                                   musaStream_t);
template void LaunchMusaTileSmallDimsKernel<Eigen::half>(
    const Eigen::half*, TileDims, int, int64_t, Eigen::half*, musaStream_t);
template void LaunchMusaTileSmallDimsKernel<bfloat16>(
    const bfloat16*, TileDims, int, int64_t, bfloat16*, musaStream_t);
template void LaunchMusaTileSmallDimsKernel<double>(const double*, TileDims,
                                                    int, int64_t, double*,
                                                    musaStream_t);
template void LaunchMusaTileSmallDimsKernel<int32>(const int32*, TileDims, int,
                                                   int64_t, int32*,
                                                   musaStream_t);
template void LaunchMusaTileSmallDimsKernel<int64>(const int64*, TileDims, int,
                                                   int64_t, int64*,
                                                   musaStream_t);
template void LaunchMusaTileSmallDimsKernel<bool>(const bool*, TileDims, int,
                                                  int64_t, bool*,
                                                  musaStream_t);

template void LaunchMusaTileLastDimKernel<float>(const float*, int64_t,
                                                 int64_t, int64_t, float*,
                                                 musaStream_t);
template void LaunchMusaTileLastDimKernel<Eigen::half>(
    const Eigen::half*, int64_t, int64_t, int64_t, Eigen::half*,
    musaStream_t);
template void LaunchMusaTileLastDimKernel<bfloat16>(
    const bfloat16*, int64_t, int64_t, int64_t, bfloat16*, musaStream_t);
template void LaunchMusaTileLastDimKernel<double>(const double*, int64_t,
                                                  int64_t, int64_t, double*,
                                                  musaStream_t);
template void LaunchMusaTileLastDimKernel<int32>(const int32*, int64_t,
                                                 int64_t, int64_t, int32*,
                                                 musaStream_t);
template void LaunchMusaTileLastDimKernel<int64>(const int64*, int64_t,
                                                 int64_t, int64_t, int64*,
                                                 musaStream_t);
template void LaunchMusaTileLastDimKernel<bool>(const bool*, int64_t, int64_t,
                                                int64_t, bool*,
                                                musaStream_t);

}  // namespace musa
}  // namespace tensorflow
