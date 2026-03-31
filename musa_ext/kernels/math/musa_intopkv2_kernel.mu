// InTopKV2 kernel for MUSA devices
// Checks if targets are in the top-k predictions

#include <float.h>
#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

__device__ __forceinline__ float LoadAsFloat(const float* p) { return *p; }

__device__ __forceinline__ float LoadAsFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ float LoadAsFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

template <typename T, typename Tidx>
__global__ void InTopKKernel(const T* predictions, const Tidx* targets,
                             bool* output, int batch_size, int num_classes,
                             int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch_size) return;

  const T* row_predictions = predictions + row * num_classes;
  Tidx target_class = targets[row];
  float target_score = LoadAsFloat(&row_predictions[target_class]);

  // Count how many classes have strictly higher score than target
  int count_higher = 0;
  for (int i = 0; i < num_classes; i++) {
    float score = LoadAsFloat(&row_predictions[i]);
    if (score > target_score) {
      count_higher++;
    }
  }

  // Target is in top-k if less than k classes have strictly higher score
  output[row] = (count_higher < k);
}

}  // namespace

// Launcher functions for int32 targets
template <typename T>
void LaunchInTopKV2Int32(const T* predictions, const int32_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (batch_size + block_size - 1) / block_size;

  InTopKKernel<T, int32_t><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, output, batch_size, num_classes, k);
}

// Launcher functions for int64 targets
template <typename T>
void LaunchInTopKV2Int64(const T* predictions, const int64_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (batch_size + block_size - 1) / block_size;

  InTopKKernel<T, int64_t><<<grid_size, block_size, 0, stream>>>(
      predictions, targets, output, batch_size, num_classes, k);
}

// Explicit instantiations for int32 targets
template void LaunchInTopKV2Int32<float>(const float*, const int32_t*, bool*, int,
                                         int, int, musaStream_t);
template void LaunchInTopKV2Int32<Eigen::half>(const Eigen::half*, const int32_t*,
                                               bool*, int, int, int, musaStream_t);
template void LaunchInTopKV2Int32<bfloat16>(const bfloat16*, const int32_t*, bool*,
                                            int, int, int, musaStream_t);

// Explicit instantiations for int64 targets
template void LaunchInTopKV2Int64<float>(const float*, const int64_t*, bool*, int,
                                         int, int, musaStream_t);
template void LaunchInTopKV2Int64<Eigen::half>(const Eigen::half*, const int64_t*,
                                               bool*, int, int, int, musaStream_t);
template void LaunchInTopKV2Int64<bfloat16>(const bfloat16*, const int64_t*, bool*,
                                            int, int, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow