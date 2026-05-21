#include <stdint.h>

#include <musa_runtime.h>

namespace {

__global__ void GreaterEqualFloatScalarKernel(const float* __restrict__ x,
                                              const float* __restrict__ y,
                                              bool* __restrict__ out,
                                              int64_t n) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const float threshold = y[0];
  for (int64_t i = tid; i < n; i += stride) {
    out[i] = x[i] >= threshold;
  }
}

}  // namespace

extern "C" void LaunchGreaterEqualFloatScalar(const float* x, const float* y,
                                               bool* out, int64_t n,
                                               musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks > 4096) blocks = 4096;
  GreaterEqualFloatScalarKernel<<<blocks, kThreads, 0, stream>>>(x, y, out, n);
}
