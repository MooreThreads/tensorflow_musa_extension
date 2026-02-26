#include <musa_runtime.h>

#include "tensorflow/core/framework/bfloat16.h"

namespace tensorflow {
namespace musa {

template <typename T>
__global__ void MusaDiagPartKernel(const int64 size, const T* __restrict__ in,
                                   T* __restrict__ out) {
  int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = idx; i < size; i += blockDim.x * gridDim.x) {
    out[i] = in[i * (size + 1)];
  }
}

template <typename T>
void MusaDiagPartkernelLauncher(musaStream_t stream, uint64_t size, const T* in,
                                T* out) {
  if (size == 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (size + block_size - 1) / block_size;
  MusaDiagPartKernel<T><<<grid_size, block_size, 0, stream>>>(size, in, out);
}

template void MusaDiagPartkernelLauncher<float>(musaStream_t, uint64_t,
                                                const float*, float*);
template void MusaDiagPartkernelLauncher<double>(musaStream_t, uint64_t,
                                                 const double*, double*);
template void MusaDiagPartkernelLauncher<int32>(musaStream_t, uint64_t,
                                                const int*, int*);
template void MusaDiagPartkernelLauncher<int64>(musaStream_t, uint64_t,
                                                const int64*, int64*);
template void MusaDiagPartkernelLauncher<Eigen::half>(musaStream_t, uint64_t,
                                                      const Eigen::half*,
                                                      Eigen::half*);
template void MusaDiagPartkernelLauncher<Eigen::bfloat16>(
    musaStream_t, uint64_t, const Eigen::bfloat16*, Eigen::bfloat16*);

}  // namespace musa
}  // namespace tensorflow