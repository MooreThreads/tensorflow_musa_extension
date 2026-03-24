#include <math.h>

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

// Warp size - use 32 for shuffle operations to be safe
// Even if MUSA supports larger warps, __shfl_xor_sync typically works within 32 threads
constexpr int kWarpSize = 32;

// Type conversion utilities
__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  uint16_t* b_ptr = (uint16_t*)p;
  uint32_t* f_ptr = (uint32_t*)&res;
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  uint32_t* f_ptr = (uint32_t*)&v;
  uint16_t b_val = (*f_ptr) >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

// Warp reduce for sum using 32-thread warp
template <int BLOCK_SIZE>
__device__ __forceinline__ float BlockReduceSum(float val, float* shared) {
  int tid = threadIdx.x;
  int lane = tid & (kWarpSize - 1);  // tid % 32
  int wid = tid >> 5;                 // tid / 32

  // Warp reduce using shuffle (32 threads)
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }

  // Write warp results to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // First warp reduces the warp results
  if (wid == 0) {
    val = (tid < (BLOCK_SIZE >> 5)) ? shared[tid] : 0.0f;
#pragma unroll
    for (int mask = (BLOCK_SIZE >> 5) >> 1; mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
  }

  return val;
}

// Normalize kernel: each block processes one row (last dimension)
// Computes: mean = sum(x) / N
//           var = sum((x - mean)^2) / N  (two-pass algorithm for numerical stability)
//           clipped_std = clamp(sqrt(var), epsilon, max_std)
//           output = (x - mean) / clipped_std
template <typename T, int BLOCK_SIZE>
__global__ void NormalizeKernel(const T* src, T* dst, int64_t row_size,
                                float epsilon, float max_std) {
  const int64_t row_idx = blockIdx.x;
  const T* row_src = src + row_idx * row_size;
  T* row_dst = dst + row_idx * row_size;

  __shared__ float shared_sum[BLOCK_SIZE / 32];  // 32-thread warp

  // First pass: compute mean
  float sum = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    sum += LoadFloat(&row_src[i]);
  }
  float total_sum = BlockReduceSum<BLOCK_SIZE>(sum, shared_sum);

  // Broadcast mean from first thread
  __shared__ float mean_shared;
  if (threadIdx.x == 0) {
    mean_shared = total_sum / static_cast<float>(row_size);
  }
  __syncthreads();

  const float mean = mean_shared;

  // Second pass: compute variance using E[(x - mean)^2]
  float var_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float val = LoadFloat(&row_src[i]);
    const float diff = val - mean;
    var_sum += diff * diff;
  }
  float total_var_sum = BlockReduceSum<BLOCK_SIZE>(var_sum, shared_sum);

  // Broadcast inv_std from first thread
  __shared__ float inv_std_shared;
  if (threadIdx.x == 0) {
    const float variance = total_var_sum / static_cast<float>(row_size);
    const float std_val = sqrtf(max(variance, 0.0f));
    // Clamp std to [epsilon, max_std]
    const float clipped_std = max(epsilon, min(std_val, max_std));
    inv_std_shared = 1.0f / clipped_std;
  }
  __syncthreads();

  const float inv_std = inv_std_shared;

  // Third pass: normalize
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float val = LoadFloat(&row_src[i]);
    const float normalized = (val - mean) * inv_std;
    StoreFloat(&row_dst[i], normalized);
  }
}

// Kernel for handling small row sizes with a single warp (32 threads)
// Uses two-pass algorithm: first compute mean, then compute E[(x-mean)^2]
template <typename T, int ROW_SIZE>
__global__ void NormalizeKernelSmall(const T* src, T* dst, int64_t num_rows,
                                     float epsilon, float max_std) {
  const int64_t row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (row_idx >= num_rows) return;

  const T* row_src = src + row_idx * ROW_SIZE;
  T* row_dst = dst + row_idx * ROW_SIZE;

  const int lane = threadIdx.x;

  // First pass: compute sum using warp reduce (32 threads)
  float sum = 0.0f;
  if (lane < ROW_SIZE) {
    sum = LoadFloat(&row_src[lane]);
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }

  const float mean = sum / static_cast<float>(ROW_SIZE);

  // Second pass: compute variance using E[(x - mean)^2]
  float var_sum = 0.0f;
  if (lane < ROW_SIZE) {
    const float val = LoadFloat(&row_src[lane]);
    const float diff = val - mean;
    var_sum = diff * diff;
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    var_sum += __shfl_xor_sync(0xffffffff, var_sum, mask);
  }

  const float variance = var_sum / static_cast<float>(ROW_SIZE);
  const float std_val = sqrtf(max(variance, 0.0f));
  // Clamp std to [epsilon, max_std]
  const float clipped_std = max(epsilon, min(std_val, max_std));
  const float inv_std = 1.0f / clipped_std;

  // Third pass: normalize
  if (lane < ROW_SIZE) {
    const float val = LoadFloat(&row_src[lane]);
    const float normalized = (val - mean) * inv_std;
    StoreFloat(&row_dst[lane], normalized);
  }
}

}  // namespace

template <typename T>
void LaunchNormalize(const T* src, T* dst, int64_t num_rows, int64_t row_size,
                     float epsilon, float max_std, musaStream_t stream) {
  if (num_rows <= 0 || row_size <= 0) return;

  const int kBlockSize = 256;

  if (row_size <= 32) {
    // Use optimized kernel for small row sizes
    // Each warp (32 threads) processes one row
    const int rows_per_block = kBlockSize / 32;
    const int64_t blocks = (num_rows + rows_per_block - 1) / rows_per_block;
    dim3 block_dim(32, rows_per_block);

    // Dispatch based on actual row size for optimal performance
    // Template parameter must match actual row_size for correct warp reduction
    switch (row_size) {
#define DISPATCH_SMALL_KERNEL(N)                                        \
      case N:                                                           \
        NormalizeKernelSmall<T, N>                                      \
            <<<blocks, block_dim, 0, stream>>>(src, dst, num_rows, epsilon, max_std); \
        break

      DISPATCH_SMALL_KERNEL(1);
      DISPATCH_SMALL_KERNEL(2);
      DISPATCH_SMALL_KERNEL(3);
      DISPATCH_SMALL_KERNEL(4);
      DISPATCH_SMALL_KERNEL(5);
      DISPATCH_SMALL_KERNEL(6);
      DISPATCH_SMALL_KERNEL(7);
      DISPATCH_SMALL_KERNEL(8);
      DISPATCH_SMALL_KERNEL(9);
      DISPATCH_SMALL_KERNEL(10);
      DISPATCH_SMALL_KERNEL(11);
      DISPATCH_SMALL_KERNEL(12);
      DISPATCH_SMALL_KERNEL(13);
      DISPATCH_SMALL_KERNEL(14);
      DISPATCH_SMALL_KERNEL(15);
      DISPATCH_SMALL_KERNEL(16);
      DISPATCH_SMALL_KERNEL(17);
      DISPATCH_SMALL_KERNEL(18);
      DISPATCH_SMALL_KERNEL(19);
      DISPATCH_SMALL_KERNEL(20);
      DISPATCH_SMALL_KERNEL(21);
      DISPATCH_SMALL_KERNEL(22);
      DISPATCH_SMALL_KERNEL(23);
      DISPATCH_SMALL_KERNEL(24);
      DISPATCH_SMALL_KERNEL(25);
      DISPATCH_SMALL_KERNEL(26);
      DISPATCH_SMALL_KERNEL(27);
      DISPATCH_SMALL_KERNEL(28);
      DISPATCH_SMALL_KERNEL(29);
      DISPATCH_SMALL_KERNEL(30);
      DISPATCH_SMALL_KERNEL(31);
      DISPATCH_SMALL_KERNEL(32);
#undef DISPATCH_SMALL_KERNEL
      default:
        // Fallback to general kernel for unexpected row sizes
        NormalizeKernel<T, kBlockSize>
            <<<num_rows, kBlockSize, 0, stream>>>(src, dst, row_size, epsilon, max_std);
        break;
    }
  } else {
    // General kernel: one block per row
    NormalizeKernel<T, kBlockSize>
        <<<num_rows, kBlockSize, 0, stream>>>(src, dst, row_size, epsilon, max_std);
  }
}

template void LaunchNormalize<float>(const float*, float*, int64_t, int64_t,
                                     float, float, musaStream_t);
template void LaunchNormalize<Eigen::half>(const Eigen::half*, Eigen::half*,
                                           int64_t, int64_t, float, float,
                                           musaStream_t);
template void LaunchNormalize<bfloat16>(const bfloat16*, bfloat16*, int64_t,
                                        int64_t, float, float, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
