#include <musa_bf16.h>
#include <musa_runtime.h>
#include <stdint.h>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = 4;
constexpr int kMaxBlocks = 4096;

static inline int64_t CeilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

static inline int ClampBlocks(int64_t items, int64_t items_per_block) {
  int64_t blocks = CeilDiv(items, items_per_block);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : static_cast<int>(blocks);
}

__global__ __launch_bounds__(kThreadsPerBlock) void FloatToBFloat16Kernel(
    const float* __restrict__ src, __mt_bfloat16* __restrict__ dst,
    int64_t n) {
  int64_t idx =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        dst[offset] = __float2bfloat16(src[offset]);
      }
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void BFloat16ToFloatKernel(
    const __mt_bfloat16* __restrict__ src, float* __restrict__ dst,
    int64_t n) {
  int64_t idx =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        dst[offset] = __bfloat162float(src[offset]);
      }
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void BoolToBFloat16Kernel(
    const bool* __restrict__ src, __mt_bfloat16* __restrict__ dst,
    int64_t n) {
  int64_t idx =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        dst[offset] = __float2bfloat16(src[offset] ? 1.0f : 0.0f);
      }
    }
  }
}

}  // namespace

template <typename DstT>
__global__ void BoolCastKernel(const bool* src, DstT* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] ? static_cast<DstT>(1) : static_cast<DstT>(0);
    }
}

template <typename DstT>
void LaunchBoolCast(const bool* src, DstT* dst, int n, musaStream_t stream) {
    if (n <= 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    BoolCastKernel<DstT><<<blocks_per_grid, threads_per_block, 0, stream>>>(src, dst, n);
}

template void LaunchBoolCast<float>(const bool*, float*, int, musaStream_t);
template void LaunchBoolCast<int32_t>(const bool*, int32_t*, int, musaStream_t);

extern "C" {

void LaunchMusaCastFloatToBFloat16(const void* src, void* dst, int64_t n,
                                   musaStream_t stream) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  FloatToBFloat16Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const float*>(src), static_cast<__mt_bfloat16*>(dst), n);
}

void LaunchMusaCastBFloat16ToFloat(const void* src, void* dst, int64_t n,
                                   musaStream_t stream) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BFloat16ToFloatKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(src), static_cast<float*>(dst), n);
}

void LaunchMusaCastBoolToBFloat16(const void* src, void* dst, int64_t n,
                                  musaStream_t stream) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BoolToBFloat16Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const bool*>(src), static_cast<__mt_bfloat16*>(dst), n);
}

}  // extern "C"

} // namespace musa
} // namespace tensorflow
