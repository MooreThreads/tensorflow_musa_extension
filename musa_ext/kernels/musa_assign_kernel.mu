// musa_assign_kernel.mu
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

template <typename T>
__global__ void AssignCopyKernel(const T* __restrict__ src, T* __restrict__ dst,
                                 int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

template <typename T>
void LaunchAssignCopy(const T* src, T* dst, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  AssignCopyKernel<T><<<blocks, kThreads, 0, stream>>>(src, dst, n);
}


template void LaunchAssignCopy<float>(const float*, float*, int64_t,
                                      musaStream_t);
template void LaunchAssignCopy<double>(const double*, double*, int64_t,
                                       musaStream_t);
template void LaunchAssignCopy<Eigen::half>(const Eigen::half*, Eigen::half*,
                                            int64_t, musaStream_t);
template void LaunchAssignCopy<bfloat16>(const bfloat16*, bfloat16*, int64_t,
                                         musaStream_t);

}  // namespace musa
}  // namespace tensorflow