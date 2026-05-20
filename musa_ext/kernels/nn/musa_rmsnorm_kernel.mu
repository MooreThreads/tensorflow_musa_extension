#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float LoadBFloat16(const tensorflow::bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreBFloat16(tensorflow::bfloat16* p,
                                              float v) {
  const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&v);
  uint16_t* b_ptr = reinterpret_cast<uint16_t*>(p);
  *b_ptr = static_cast<uint16_t>(*f_ptr >> 16);
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float BlockReduceSum(float val, float* shared) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int wid = tid >> 5;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (tid < (BLOCK_SIZE >> 5)) ? shared[lane] : 0.0f;
#pragma unroll
    for (int mask = (BLOCK_SIZE >> 6); mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
  }

  return val;
}

template <int BLOCK_SIZE>
__global__ void RmsNormFloatKernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ y,
                                   float* __restrict__ normalized,
                                   float* __restrict__ inv_rms,
                                   int64_t row_size, float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * row_size;

  __shared__ float shared[BLOCK_SIZE / kWarpSize];

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float v = x[base + i];
    sum_sq += v * v;
  }

  const float total_sq = BlockReduceSum<BLOCK_SIZE>(sum_sq, shared);

  __shared__ float inv_shared;
  if (threadIdx.x == 0) {
    inv_shared = rsqrtf(total_sq / static_cast<float>(row_size) + epsilon);
    inv_rms[row] = inv_shared;
  }
  __syncthreads();

  const float inv = inv_shared;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float norm = x[base + i] * inv;
    normalized[base + i] = norm;
    y[base + i] = norm * gamma[i];
  }
}


template <int ROWS_PER_BLOCK>
__global__ void RmsNormFloatRow128Kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         float* __restrict__ y,
                                         float* __restrict__ normalized,
                                         float* __restrict__ inv_rms,
                                         int64_t num_rows, float epsilon) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (row >= num_rows) return;

  constexpr int kRowSize = 128;
  const int64_t base = row * kRowSize;

  float sum_sq = 0.0f;
#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const float v = x[base + i];
    sum_sq += v * v;
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
  }

  const float inv = rsqrtf(sum_sq * (1.0f / 128.0f) + epsilon);
  if (lane == 0) {
    inv_rms[row] = inv;
  }

#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const float norm = x[base + i] * inv;
    normalized[base + i] = norm;
    y[base + i] = norm * gamma[i];
  }
}


template <int ROWS_PER_BLOCK>
__global__ void RmsNormGradDxRow128Kernel(const float* __restrict__ x,
                                          const float* __restrict__ inv_rms,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ dy,
                                          float* __restrict__ dx,
                                          int64_t num_rows) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (row >= num_rows) return;

  constexpr int kRowSize = 128;
  const int64_t base = row * kRowSize;

  float dot = 0.0f;
#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    dot += x[base + i] * dy[base + i] * gamma[i];
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    dot += __shfl_xor_sync(0xffffffff, dot, mask);
  }

  const float inv = inv_rms[row];
  const float coeff = dot * inv * inv * inv * (1.0f / 128.0f);

#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const int64_t idx = base + i;
    dx[idx] = dy[idx] * gamma[i] * inv - x[idx] * coeff;
  }
}

template <int BLOCK_SIZE>
__global__ void RmsNormGradDxKernel(const float* __restrict__ x,
                                    const float* __restrict__ inv_rms,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ dy,
                                    float* __restrict__ dx,
                                    int64_t row_size) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * row_size;
  __shared__ float shared[BLOCK_SIZE / kWarpSize];

  float dot = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    dot += x[base + i] * dy[base + i] * gamma[i];
  }
  const float total_dot = BlockReduceSum<BLOCK_SIZE>(dot, shared);

  __shared__ float coeff_shared;
  __shared__ float inv_shared;
  if (threadIdx.x == 0) {
    const float inv = inv_rms[row];
    inv_shared = inv;
    coeff_shared = total_dot * inv * inv * inv /
                   static_cast<float>(row_size);
  }
  __syncthreads();

  const float inv = inv_shared;
  const float coeff = coeff_shared;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const int64_t idx = base + i;
    dx[idx] = dy[idx] * gamma[i] * inv - x[idx] * coeff;
  }
}

template <int BLOCK_SIZE>
__global__ void RmsNormBFloat16Kernel(
    const tensorflow::bfloat16* __restrict__ x,
    const tensorflow::bfloat16* __restrict__ gamma,
    tensorflow::bfloat16* __restrict__ y,
    tensorflow::bfloat16* __restrict__ normalized,
    tensorflow::bfloat16* __restrict__ inv_rms, int64_t row_size,
    float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * row_size;
  __shared__ float shared[BLOCK_SIZE / kWarpSize];

  float sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float v = LoadBFloat16(&x[base + i]);
    sum_sq += v * v;
  }
  const float total_sq = BlockReduceSum<BLOCK_SIZE>(sum_sq, shared);

  __shared__ float inv_shared;
  if (threadIdx.x == 0) {
    inv_shared = rsqrtf(total_sq / static_cast<float>(row_size) + epsilon);
    StoreBFloat16(&inv_rms[row], inv_shared);
  }
  __syncthreads();

  const float inv = inv_shared;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const float norm = LoadBFloat16(&x[base + i]) * inv;
    StoreBFloat16(&normalized[base + i], norm);
    StoreBFloat16(&y[base + i], norm * LoadBFloat16(&gamma[i]));
  }
}

template <int ROWS_PER_BLOCK>
__global__ void RmsNormBFloat16Row128Kernel(
    const tensorflow::bfloat16* __restrict__ x,
    const tensorflow::bfloat16* __restrict__ gamma,
    tensorflow::bfloat16* __restrict__ y,
    tensorflow::bfloat16* __restrict__ normalized,
    tensorflow::bfloat16* __restrict__ inv_rms, int64_t num_rows,
    float epsilon) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (row >= num_rows) return;

  constexpr int kRowSize = 128;
  const int64_t base = row * kRowSize;

  float sum_sq = 0.0f;
#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const float v = LoadBFloat16(&x[base + i]);
    sum_sq += v * v;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
  }

  const float inv = rsqrtf(sum_sq * (1.0f / 128.0f) + epsilon);
  if (lane == 0) StoreBFloat16(&inv_rms[row], inv);

#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const float norm = LoadBFloat16(&x[base + i]) * inv;
    StoreBFloat16(&normalized[base + i], norm);
    StoreBFloat16(&y[base + i], norm * LoadBFloat16(&gamma[i]));
  }
}

template <int ROWS_PER_BLOCK>
__global__ void RmsNormGradDxBFloat16Row128Kernel(
    const tensorflow::bfloat16* __restrict__ x,
    const tensorflow::bfloat16* __restrict__ inv_rms,
    const tensorflow::bfloat16* __restrict__ gamma,
    const tensorflow::bfloat16* __restrict__ dy,
    tensorflow::bfloat16* __restrict__ dx, int64_t num_rows) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (row >= num_rows) return;

  constexpr int kRowSize = 128;
  const int64_t base = row * kRowSize;

  float dot = 0.0f;
#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    dot += LoadBFloat16(&x[base + i]) * LoadBFloat16(&dy[base + i]) *
           LoadBFloat16(&gamma[i]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    dot += __shfl_xor_sync(0xffffffff, dot, mask);
  }

  const float inv = LoadBFloat16(&inv_rms[row]);
  const float coeff = dot * inv * inv * inv * (1.0f / 128.0f);

#pragma unroll
  for (int i = lane; i < kRowSize; i += 32) {
    const int64_t idx = base + i;
    StoreBFloat16(&dx[idx], LoadBFloat16(&dy[idx]) *
                                LoadBFloat16(&gamma[i]) * inv -
                            LoadBFloat16(&x[idx]) * coeff);
  }
}

template <int BLOCK_SIZE>
__global__ void RmsNormGradDxBFloat16Kernel(
    const tensorflow::bfloat16* __restrict__ x,
    const tensorflow::bfloat16* __restrict__ inv_rms,
    const tensorflow::bfloat16* __restrict__ gamma,
    const tensorflow::bfloat16* __restrict__ dy,
    tensorflow::bfloat16* __restrict__ dx, int64_t row_size) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * row_size;
  __shared__ float shared[BLOCK_SIZE / kWarpSize];

  float dot = 0.0f;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    dot += LoadBFloat16(&x[base + i]) * LoadBFloat16(&dy[base + i]) *
           LoadBFloat16(&gamma[i]);
  }
  const float total_dot = BlockReduceSum<BLOCK_SIZE>(dot, shared);

  __shared__ float coeff_shared;
  __shared__ float inv_shared;
  if (threadIdx.x == 0) {
    const float inv = LoadBFloat16(&inv_rms[row]);
    inv_shared = inv;
    coeff_shared = total_dot * inv * inv * inv /
                   static_cast<float>(row_size);
  }
  __syncthreads();

  const float inv = inv_shared;
  const float coeff = coeff_shared;
  for (int64_t i = threadIdx.x; i < row_size; i += BLOCK_SIZE) {
    const int64_t idx = base + i;
    StoreBFloat16(&dx[idx], LoadBFloat16(&dy[idx]) *
                                LoadBFloat16(&gamma[i]) * inv -
                            LoadBFloat16(&x[idx]) * coeff);
  }
}


}  // namespace

extern "C" void LaunchRmsNormFloat(const float* x, const float* gamma,
                                    float* y, float* normalized,
                                    float* inv_rms, int64_t num_rows,
                                    int64_t row_size, float epsilon,
                                    musaStream_t stream) {
  if (num_rows <= 0 || row_size <= 0) return;
  if (row_size == 128) {
    constexpr int kRowsPerBlock = 8;
    dim3 block(32, kRowsPerBlock);
    const unsigned int blocks =
        static_cast<unsigned int>((num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    RmsNormFloatRow128Kernel<kRowsPerBlock>
        <<<blocks, block, 0, stream>>>(x, gamma, y, normalized, inv_rms,
                                       num_rows, epsilon);
    return;
  }

  constexpr int kBlockSize = 256;
  RmsNormFloatKernel<kBlockSize>
      <<<static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
          x, gamma, y, normalized, inv_rms, row_size, epsilon);
}


extern "C" void LaunchRmsNormGradDxFloat(const float* x,
                                          const float* inv_rms,
                                          const float* gamma,
                                          const float* dy, float* dx,
                                          int64_t num_rows,
                                          int64_t row_size,
                                          musaStream_t stream) {
  if (num_rows <= 0 || row_size <= 0) return;
  if (row_size == 128) {
    constexpr int kRowsPerBlock = 8;
    dim3 block(32, kRowsPerBlock);
    const unsigned int blocks =
        static_cast<unsigned int>((num_rows + kRowsPerBlock - 1) /
                                  kRowsPerBlock);
    RmsNormGradDxRow128Kernel<kRowsPerBlock>
        <<<blocks, block, 0, stream>>>(x, inv_rms, gamma, dy, dx, num_rows);
    return;
  }

  constexpr int kBlockSize = 256;
  RmsNormGradDxKernel<kBlockSize>
      <<<static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
          x, inv_rms, gamma, dy, dx, row_size);
}

extern "C" void LaunchRmsNormBFloat16(
    const tensorflow::bfloat16* x, const tensorflow::bfloat16* gamma,
    tensorflow::bfloat16* y, tensorflow::bfloat16* normalized,
    tensorflow::bfloat16* inv_rms, int64_t num_rows, int64_t row_size,
    float epsilon, musaStream_t stream) {
  if (num_rows <= 0 || row_size <= 0) return;
  if (row_size == 128) {
    constexpr int kRowsPerBlock = 8;
    dim3 block(32, kRowsPerBlock);
    const unsigned int blocks =
        static_cast<unsigned int>((num_rows + kRowsPerBlock - 1) /
                                  kRowsPerBlock);
    RmsNormBFloat16Row128Kernel<kRowsPerBlock>
        <<<blocks, block, 0, stream>>>(x, gamma, y, normalized, inv_rms,
                                       num_rows, epsilon);
    return;
  }

  constexpr int kBlockSize = 256;
  RmsNormBFloat16Kernel<kBlockSize>
      <<<static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
          x, gamma, y, normalized, inv_rms, row_size, epsilon);
}

extern "C" void LaunchRmsNormGradDxBFloat16(
    const tensorflow::bfloat16* x, const tensorflow::bfloat16* inv_rms,
    const tensorflow::bfloat16* gamma, const tensorflow::bfloat16* dy,
    tensorflow::bfloat16* dx, int64_t num_rows, int64_t row_size,
    musaStream_t stream) {
  if (num_rows <= 0 || row_size <= 0) return;
  if (row_size == 128) {
    constexpr int kRowsPerBlock = 8;
    dim3 block(32, kRowsPerBlock);
    const unsigned int blocks =
        static_cast<unsigned int>((num_rows + kRowsPerBlock - 1) /
                                  kRowsPerBlock);
    RmsNormGradDxBFloat16Row128Kernel<kRowsPerBlock>
        <<<blocks, block, 0, stream>>>(x, inv_rms, gamma, dy, dx, num_rows);
    return;
  }

  constexpr int kBlockSize = 256;
  RmsNormGradDxBFloat16Kernel<kBlockSize>
      <<<static_cast<unsigned int>(num_rows), kBlockSize, 0, stream>>>(
          x, inv_rms, gamma, dy, dx, row_size);
}
