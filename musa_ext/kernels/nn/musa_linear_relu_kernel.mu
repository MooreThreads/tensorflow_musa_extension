#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
template <typename T>
__global__ void BiasAddReluKernel(const T* x, const T* bias, T* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val = x[idx] + bias[idx];
    output[idx] = val > static_cast<T>(0) ? val : static_cast<T>(0);
  }
}

template <typename T>
void LaunchBiasAddReluKernel(const T* x, const T* bias, T* output, int n,
                             musaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  BiasAddReluKernel<T>
      <<<num_blocks, block_size, 0, stream>>>(x, bias, output, n);
}
}  // namespace musa
}  // namespace tensorflow