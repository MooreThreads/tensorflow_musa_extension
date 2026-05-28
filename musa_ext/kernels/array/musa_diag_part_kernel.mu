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
                                                const int32*, int32*);
template void MusaDiagPartkernelLauncher<long long>(musaStream_t, uint64_t,
                                                    const long long*,
                                                    long long*);
template void MusaDiagPartkernelLauncher<long>(musaStream_t, uint64_t,
                                               const long*,
                                               long*);
template void MusaDiagPartkernelLauncher<Eigen::half>(musaStream_t, uint64_t,
                                                      const Eigen::half*,
                                                      Eigen::half*);
template void MusaDiagPartkernelLauncher<Eigen::bfloat16>(
    musaStream_t, uint64_t, const Eigen::bfloat16*, Eigen::bfloat16*);

// ==========================================
// MatrixDiagPartV3 kernel
// ==========================================
// Handles batched [..., M, N] inputs with diagonal offset k.
// For k >= 0: extracts super-diagonal k; for k < 0: sub-diagonal.
// Supports both scalar k and range k = [k_min, k_max] (multiple diagonals).
template <typename T>
__global__ void MusaMatrixDiagPartV3Kernel(
    const T* __restrict__ input, T* __restrict__ output,
    const T padding_value,
    int64 batch_size, int64 M, int64 N,
    int k_min, int k_max,
    int64 num_diags, int64 max_diag_len) {
  int64 tid = (int64)blockIdx.x * blockDim.x + threadIdx.x;
  int64 total = batch_size * num_diags * max_diag_len;
  if (tid >= total) return;

  int64 b = tid / (num_diags * max_diag_len);
  int64 rem = tid % (num_diags * max_diag_len);
  int64 d = rem / max_diag_len;  // d=0 => k_max diagonal
  int64 i = rem % max_diag_len;

  int k = k_max - (int)d;
  int64 k_abs = (k < 0) ? (int64)(-k) : (int64)k;
  int64 diag_len = (M < N ? M : N) - k_abs;

  if (diag_len <= 0 || i >= diag_len) {
    output[tid] = padding_value;
    return;
  }

  int64 row = i + (k < 0 ? k_abs : 0LL);
  int64 col = i + (k > 0 ? (int64)k : 0LL);
  output[tid] = input[b * M * N + row * N + col];
}

template <typename T>
void MusaMatrixDiagPartV3KernelLauncher(musaStream_t stream, int64 batch_size,
                                         int64 M, int64 N, int k_min, int k_max,
                                         int64 num_diags, int64 max_diag_len,
                                         const T padding_value, const T* input,
                                         T* output) {
  int64 total = batch_size * num_diags * max_diag_len;
  if (total == 0) return;
  const int block_size = 256;
  const int grid_size =
      static_cast<int>((total + block_size - 1) / block_size);
  MusaMatrixDiagPartV3Kernel<T><<<grid_size, block_size, 0, stream>>>(
      input, output, padding_value, batch_size, M, N, k_min, k_max, num_diags,
      max_diag_len);
}

template void MusaMatrixDiagPartV3KernelLauncher<float>(musaStream_t, int64,
    int64, int64, int, int, int64, int64, const float, const float*, float*);
template void MusaMatrixDiagPartV3KernelLauncher<double>(musaStream_t, int64,
    int64, int64, int, int, int64, int64, const double, const double*, double*);
template void MusaMatrixDiagPartV3KernelLauncher<int32>(musaStream_t, int64,
    int64, int64, int, int, int64, int64, const int32, const int32*, int32*);
template void MusaMatrixDiagPartV3KernelLauncher<long long>(musaStream_t, int64,
    int64, int64, int, int, int64, int64, const long long, const long long*,
    long long*);
template void MusaMatrixDiagPartV3KernelLauncher<long>(musaStream_t, int64,
    int64, int64, int, int, int64, int64, const long, const long*, long*);
template void MusaMatrixDiagPartV3KernelLauncher<Eigen::half>(musaStream_t,
    int64, int64, int64, int, int, int64, int64, const Eigen::half,
    const Eigen::half*, Eigen::half*);
template void MusaMatrixDiagPartV3KernelLauncher<Eigen::bfloat16>(musaStream_t,
    int64, int64, int64, int, int, int64, int64, const Eigen::bfloat16,
    const Eigen::bfloat16*, Eigen::bfloat16*);

}  // namespace musa
}  // namespace tensorflow
