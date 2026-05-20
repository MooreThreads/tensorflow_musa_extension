#include <musa_bf16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;

template <int RowsPerBlock, int ThreadsPerRow>
__global__ __launch_bounds__(kThreadsPerBlock)
void MeanLastDimBFloat16SmallReduceKernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_bfloat16* __restrict__ output, int64_t rows, int64_t reduce) {
  __shared__ float shared[kThreadsPerBlock];

  const int tid = threadIdx.x;
  const int row_in_block = tid / ThreadsPerRow;
  const int lane = tid - row_in_block * ThreadsPerRow;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * RowsPerBlock + row_in_block;

  float sum = 0.0f;
  if (row < rows) {
    const int64_t base = row * reduce;
    for (int64_t i = lane; i < reduce; i += ThreadsPerRow) {
      sum += __bfloat162float(input[base + i]);
    }
  }

  shared[tid] = sum;
  __syncthreads();

  const int shared_base = row_in_block * ThreadsPerRow;
  for (int stride = ThreadsPerRow / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared[shared_base + lane] += shared[shared_base + lane + stride];
    }
    __syncthreads();
  }

  if (row < rows && lane == 0) {
    output[row] = __float2bfloat16(shared[shared_base] /
                                   static_cast<float>(reduce));
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MeanLastDimBFloat16Kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_bfloat16* __restrict__ output, int64_t rows, int64_t reduce) {
  __shared__ float shared[kThreadsPerBlock];

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  const int64_t base = row * reduce;
  for (int64_t i = tid; i < reduce; i += blockDim.x) {
    sum += __bfloat162float(input[base + i]);
  }

  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[row] = __float2bfloat16(shared[0] / static_cast<float>(reduce));
  }
}

}  // namespace

extern "C" void LaunchMeanLastDimBFloat16(const void* input, void* output,
                                          int64_t rows, int64_t reduce,
                                          musaStream_t stream) {
  if (rows <= 0 || reduce <= 0) return;
  if (reduce <= 64) {
    constexpr int kRowsPerBlock = 8;
    const int64_t blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    MeanLastDimBFloat16SmallReduceKernel<8, 32>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(input),
            static_cast<__mt_bfloat16*>(output), rows, reduce);
    return;
  }
  if (reduce <= 128) {
    constexpr int kRowsPerBlock = 4;
    const int64_t blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    MeanLastDimBFloat16SmallReduceKernel<4, 64>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(input),
            static_cast<__mt_bfloat16*>(output), rows, reduce);
    return;
  }
  if (reduce <= 256) {
    constexpr int kRowsPerBlock = 2;
    const int64_t blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    MeanLastDimBFloat16SmallReduceKernel<2, 128>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(input),
            static_cast<__mt_bfloat16*>(output), rows, reduce);
    return;
  }
  MeanLastDimBFloat16Kernel<<<rows, kThreadsPerBlock, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<__mt_bfloat16*>(output), rows, reduce);
}

}  // namespace musa
}  // namespace tensorflow
