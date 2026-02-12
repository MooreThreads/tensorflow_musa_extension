#include <musa_runtime.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

template <typename T>
__global__ void AssignCopyKernel(const T* __restrict__ src,
                                 T* __restrict__ dst,
                                 int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

template <typename T>
void LaunchAssignCopy(const T* src, T* dst, int64_t n, musaStream_t stream) {
  if (n <= 0) return;

  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  AssignCopyKernel<T><<<blocks, threads, 0, stream>>>(src, dst, n);

  // 与你 addn_kernel.mu 保持一致：这里不强制报错，由上层统一处理也行
  musaError_t err = musaGetLastError();
  if (err != musaSuccess) {
    // 这里留空：上层可选择做 OP_REQUIRES 检查
  }
}

// 显式实例化（只覆盖你要求的 4 种类型）
template void LaunchAssignCopy<float>(const float*, float*, int64_t, musaStream_t);
template void LaunchAssignCopy<double>(const double*, double*, int64_t, musaStream_t);
template void LaunchAssignCopy<Eigen::half>(const Eigen::half*, Eigen::half*, int64_t, musaStream_t);
template void LaunchAssignCopy<bfloat16>(const bfloat16*, bfloat16*, int64_t, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
