#include <musa_runtime.h>
#include <stdint.h> 

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#pragma GCC diagnostic pop

using bfloat16 = tensorflow::bfloat16;

namespace tensorflow {
namespace musa {

template <typename T>
__device__ __forceinline__ T DeviceNeg(T val) {
  return -val;
}

template <>
__device__ __forceinline__ Eigen::half DeviceNeg<Eigen::half>(Eigen::half val) {
  uint16_t raw = *reinterpret_cast<const uint16_t*>(&val);
  raw ^= 0x8000;
  return *reinterpret_cast<const Eigen::half*>(&raw);
}

template <>
__device__ __forceinline__ bfloat16 DeviceNeg<bfloat16>(bfloat16 val) {
  uint16_t raw = *reinterpret_cast<const uint16_t*>(&val);
  raw ^= 0x8000;
  return *reinterpret_cast<const bfloat16*>(&raw);
}

template <typename T>
__global__ void NegKernel(const T* in, T* out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = DeviceNeg(in[idx]);
  }
}

template <typename T>
void MusaNegKernelLauncher(const void* in, void* out, int size, musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (size + block_size - 1) / block_size;

  NegKernel<T><<<grid_size, block_size, 0, stream>>>(
      static_cast<const T*>(in), static_cast<T*>(out), size);
}

template void MusaNegKernelLauncher<float>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<double>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<int>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<long long>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<Eigen::half>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<bfloat16>(const void*, void*, int, musaStream_t);

} // namespace musa
} // namespace tensorflow

