#include <stdint.h>

#include <musa_bf16.h>
#include <musa_runtime.h>

namespace {

__global__ void FloatToBFloat16CopyKernel(const float* __restrict__ src,
                                          __mt_bfloat16* __restrict__ dst,
                                          int64_t n) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

}  // namespace

extern "C" void LaunchFloatToBFloat16Copy(const float* src, void* dst,
                                           int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks > 4096) blocks = 4096;
  FloatToBFloat16CopyKernel<<<blocks, kThreads, 0, stream>>>(
      src, reinterpret_cast<__mt_bfloat16*>(dst), n);
}
