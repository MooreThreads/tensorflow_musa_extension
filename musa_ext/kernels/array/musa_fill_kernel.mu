#include <musa_runtime.h>
#include <stdint.h>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;

inline int64_t CeilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename T>
__global__ void FillKernel(T* out, T value, int64_t n) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = value;
  }
}

template <typename T>
void LaunchFill(T* out, T value, int64_t n, musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  const int64_t blocks = CeilDiv(n, kThreadsPerBlock);
  FillKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(out, value, n);
}

}  // namespace

extern "C" {

void LaunchMusaFill_float(float* out, float value, int64_t n,
                          musaStream_t stream) {
  LaunchFill<float>(out, value, n, stream);
}

void LaunchMusaFill_double(double* out, double value, int64_t n,
                           musaStream_t stream) {
  LaunchFill<double>(out, value, n, stream);
}

void LaunchMusaFill_int32(int32* out, int32 value, int64_t n,
                          musaStream_t stream) {
  LaunchFill<int32>(out, value, n, stream);
}

void LaunchMusaFill_int64(int64* out, int64 value, int64_t n,
                          musaStream_t stream) {
  LaunchFill<int64>(out, value, n, stream);
}

void LaunchMusaFill_half(Eigen::half* out, Eigen::half value, int64_t n,
                         musaStream_t stream) {
  LaunchFill<Eigen::half>(out, value, n, stream);
}

void LaunchMusaFill_bfloat16(Eigen::bfloat16* out, Eigen::bfloat16 value,
                             int64_t n, musaStream_t stream) {
  LaunchFill<Eigen::bfloat16>(out, value, n, stream);
}

void LaunchMusaFill_bool(bool* out, bool value, int64_t n, musaStream_t stream) {
  LaunchFill<bool>(out, value, n, stream);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
