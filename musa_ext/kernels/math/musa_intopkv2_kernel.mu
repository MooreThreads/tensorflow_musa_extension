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

constexpr int kWarpSize = 32;
constexpr int kInTopKBlockSize = 256;

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

template <int BLOCK_SIZE>
__device__ __forceinline__ int BlockReduceSum(int val, int* shared) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int wid = tid >> 5;

#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (tid < (BLOCK_SIZE >> 5)) ? shared[lane] : 0;
#pragma unroll
    for (int mask = (BLOCK_SIZE >> 6); mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
  }

  return val;
}

template <typename T, typename Tidx, int BLOCK_SIZE>
__global__ void InTopKKernel(const T* __restrict__ predictions,
                             const Tidx* __restrict__ targets,
                             bool* __restrict__ output, int batch_size,
                             int num_classes, int k) {
  const int row = blockIdx.x;
  if (row >= batch_size) return;

  const T* row_predictions = predictions + row * num_classes;
  const Tidx target_class = targets[row];
  const float target_score = LoadAsFloat(&row_predictions[target_class]);

  int count_higher = 0;
  for (int i = threadIdx.x; i < num_classes && count_higher < k; i += BLOCK_SIZE) {
    const float score = LoadAsFloat(&row_predictions[i]);
    count_higher += score > target_score;
  }

  __shared__ int shared[BLOCK_SIZE / kWarpSize];
  const int total_count = BlockReduceSum<BLOCK_SIZE>(count_higher, shared);
  if (threadIdx.x == 0) {
    output[row] = (total_count < k);
  }
}

__global__ void SetBoolKernel(bool* output, int size, bool value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = value;
  }
}

void LaunchSetBool(bool* output, int size, bool value, musaStream_t stream) {
  constexpr int block_size = 256;
  const int grid_size = (size + block_size - 1) / block_size;
  SetBoolKernel<<<grid_size, block_size, 0, stream>>>(output, size, value);
}

}  // namespace

// Launcher functions for int32 targets
template <typename T>
void LaunchInTopKV2Int32(const T* predictions, const int32_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream) {
  if (k == 0 || k == num_classes) {
    LaunchSetBool(output, batch_size, k == num_classes, stream);
    return;
  }

  InTopKKernel<T, int32_t, kInTopKBlockSize>
      <<<batch_size, kInTopKBlockSize, 0, stream>>>(predictions, targets, output,
                                                    batch_size, num_classes, k);
}

// Launcher functions for int64 targets
template <typename T>
void LaunchInTopKV2Int64(const T* predictions, const int64_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream) {
  if (k == 0 || k == num_classes) {
    LaunchSetBool(output, batch_size, k == num_classes, stream);
    return;
  }

  InTopKKernel<T, int64_t, kInTopKBlockSize>
      <<<batch_size, kInTopKBlockSize, 0, stream>>>(predictions, targets, output,
                                                    batch_size, num_classes, k);
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