#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {
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

}  // namespace

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

}  // namespace musa
}  // namespace tensorflow
