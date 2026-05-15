#include <musa_runtime.h>
#include <stdint.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace musa {

// Device-safe zero helper.
// Eigen::half and Eigen::bfloat16 constructors from float are __host__-only,
// so we use bit-cast from uint16_t(0) instead (zero is 0x0000 for both types).
template <typename T>
__device__ __forceinline__ T DeviceZero() {
  return T(0);
}

template <>
__device__ __forceinline__ Eigen::half DeviceZero<Eigen::half>() {
  uint16_t z = 0;
  return *reinterpret_cast<const Eigen::half*>(&z);
}

template <>
__device__ __forceinline__ Eigen::bfloat16 DeviceZero<Eigen::bfloat16>() {
  uint16_t z = 0;
  return *reinterpret_cast<const Eigen::bfloat16*>(&z);
}

// Each thread handles one element of the output tensor (flattened across
// batch * m * n). Elements outside the band [num_lower, num_upper] are zeroed,
// elements inside are copied from input.
template <typename Scalar>
__global__ void MatrixBandPartKernel(const int num_threads, const int m,
                                     const int n, const int num_lower_diags,
                                     const int num_upper_diags,
                                     const Scalar* __restrict__ input_ptr,
                                     Scalar* __restrict__ output_ptr) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (; index < num_threads; index += blockDim.x * gridDim.x) {
    const int col = index % n;
    const int row = (index / n) % m;
    const int band_start =
        (num_lower_diags < 0 ? 0 : row - num_lower_diags);
    const int band_end =
        (num_upper_diags < 0 ? n : row + num_upper_diags + 1);
    if (col < band_start || col >= band_end) {
      output_ptr[index] = DeviceZero<Scalar>();
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
void MusaMatrixBandPartKernelLauncher(musaStream_t stream, const int batch_size,
                                      const int m, const int n,
                                      const int num_lower_diags,
                                      const int num_upper_diags,
                                      const Scalar* input_ptr,
                                      Scalar* output_ptr) {
  const int num_threads = batch_size * m * n;
  if (num_threads == 0) return;
  const int block_size = 256;
  const int grid_size = (num_threads + block_size - 1) / block_size;
  MatrixBandPartKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
      num_threads, m, n, num_lower_diags, num_upper_diags, input_ptr,
      output_ptr);
}

#define DEFINE_LAUNCHER(T)                                               \
  template void MusaMatrixBandPartKernelLauncher<T>(                    \
      musaStream_t, const int, const int, const int, const int,         \
      const int, const T*, T*);

DEFINE_LAUNCHER(float)
DEFINE_LAUNCHER(double)
DEFINE_LAUNCHER(int32)
DEFINE_LAUNCHER(int64_t)
DEFINE_LAUNCHER(Eigen::half)
DEFINE_LAUNCHER(Eigen::bfloat16)

#undef DEFINE_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
