#include <musa_bf16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;

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
  MeanLastDimBFloat16Kernel<<<rows, kThreadsPerBlock, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<__mt_bfloat16*>(output), rows, reduce);
}

}  // namespace musa
}  // namespace tensorflow
