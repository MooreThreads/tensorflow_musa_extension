#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

template <typename T>
__device__ __forceinline__ bool DeviceIsNan(T v) {
  // IEEE 754 property: NaN is the only value that is not equal to itself.
  return v != v;
}

template <typename T>
__global__ void IsNanKernel(const T* input, bool* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = DeviceIsNan(input[idx]);
  }
}

template <typename T>
void LaunchIsNan(const T* input, bool* output, int n, musaStream_t stream) {
  if (n == 0) return;
  // Standard 1D launch for elementwise kernels.
  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  IsNanKernel<T><<<grid_size, block_size, 0, stream>>>(input, output, n);
}

template void LaunchIsNan<Eigen::half>(const Eigen::half*, bool*, int,
                                       musaStream_t);
template void LaunchIsNan<float>(const float*, bool*, int, musaStream_t);
template void LaunchIsNan<double>(const double*, bool*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
