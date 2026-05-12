#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = 4;
constexpr int kMaxBlocks = 8192;
constexpr int kMaxGridY = 65535;

static inline int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

static inline int ClampBlocks(int64_t items, int64_t items_per_block) {
  int64_t blocks = CeilDiv(items, items_per_block);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : static_cast<int>(blocks);
}

static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__global__ __launch_bounds__(kThreadsPerBlock) void SelectSameShapeFloat4Kernel(
    const bool* __restrict__ cond, const float4* __restrict__ then_t,
    const float4* __restrict__ else_t, float4* __restrict__ out,
    int64_t vec_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const int64_t base = idx * 4;
    const float4 t = then_t[idx];
    const float4 e = else_t[idx];
    out[idx] =
        make_float4(cond[base] ? t.x : e.x, cond[base + 1] ? t.y : e.y,
                    cond[base + 2] ? t.z : e.z, cond[base + 3] ? t.w : e.w);
  }
}

__global__
__launch_bounds__(kThreadsPerBlock) void SelectScalarCondFloat4Kernel(
    const bool* __restrict__ cond, const float4* __restrict__ then_t,
    const float4* __restrict__ else_t, float4* __restrict__ out,
    int64_t vec_n) {
  const float4* __restrict__ src = cond[0] ? then_t : else_t;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    out[idx] = src[idx];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void SelectRank1CondFloat4Kernel(
    const bool* __restrict__ cond, const float4* __restrict__ then_t,
    const float4* __restrict__ else_t, float4* __restrict__ out, int64_t rows,
    int64_t inner_vecs) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const float4* __restrict__ src = cond[row] ? then_t : else_t;
    const int64_t row_offset = row * inner_vecs;
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; col < inner_vecs; col += stride) {
      const int64_t offset = row_offset + col;
      out[offset] = src[offset];
    }
  }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock) void SelectSameShapeKernel(
    const bool* __restrict__ cond, const T* __restrict__ then_t,
    const T* __restrict__ else_t, T* __restrict__ out, int64_t n) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = cond[offset] ? then_t[offset] : else_t[offset];
      }
    }
  }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock) void SelectScalarCondKernel(
    const bool* __restrict__ cond, const T* __restrict__ then_t,
    const T* __restrict__ else_t, T* __restrict__ out, int64_t n) {
  const bool choose_then = cond[0];
  const T* __restrict__ src = choose_then ? then_t : else_t;
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = src[offset];
      }
    }
  }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock) void SelectRank1CondKernel(
    const bool* __restrict__ cond, const T* __restrict__ then_t,
    const T* __restrict__ else_t, T* __restrict__ out, int64_t n,
    int64_t inner_size) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        const int64_t outer = offset / inner_size;
        out[offset] = cond[outer] ? then_t[offset] : else_t[offset];
      }
    }
  }
}

template <typename T>
void LaunchSelectSameShapeTyped(const void* cond, const void* then_t,
                                const void* else_t, void* out, int64_t n,
                                musaStream_t stream) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  SelectSameShapeKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const bool*>(cond), static_cast<const T*>(then_t),
      static_cast<const T*>(else_t), static_cast<T*>(out), n);
}

template <typename T>
void LaunchSelectScalarCondTyped(const void* cond, const void* then_t,
                                 const void* else_t, void* out, int64_t n,
                                 musaStream_t stream) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  SelectScalarCondKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const bool*>(cond), static_cast<const T*>(then_t),
      static_cast<const T*>(else_t), static_cast<T*>(out), n);
}

template <typename T>
void LaunchSelectRank1CondTyped(const void* cond, const void* then_t,
                                const void* else_t, void* out, int64_t n,
                                int64_t inner_size, musaStream_t stream) {
  if (n <= 0 || inner_size <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  SelectRank1CondKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      static_cast<const bool*>(cond), static_cast<const T*>(then_t),
      static_cast<const T*>(else_t), static_cast<T*>(out), n, inner_size);
}

}  // namespace

extern "C" {

void LaunchMusaSelectSameShapeFloat(const void* cond, const void* then_t,
                                    const void* else_t, void* out, int64_t n,
                                    musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(then_t) && IsAligned16(else_t) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    SelectSameShapeFloat4Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<const bool*>(cond), reinterpret_cast<const float4*>(then_t),
        reinterpret_cast<const float4*>(else_t), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchSelectSameShapeTyped<float>(cond, then_t, else_t, out, n, stream);
}

void LaunchMusaSelectScalarCondFloat(const void* cond, const void* then_t,
                                     const void* else_t, void* out, int64_t n,
                                     musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(then_t) && IsAligned16(else_t) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    SelectScalarCondFloat4Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<const bool*>(cond), reinterpret_cast<const float4*>(then_t),
        reinterpret_cast<const float4*>(else_t), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchSelectScalarCondTyped<float>(cond, then_t, else_t, out, n, stream);
}

void LaunchMusaSelectRank1CondFloat(const void* cond, const void* then_t,
                                    const void* else_t, void* out, int64_t n,
                                    int64_t inner_size, musaStream_t stream) {
  if (n >= 4 && (inner_size % 4) == 0 && IsAligned16(then_t) &&
      IsAligned16(else_t) && IsAligned16(out)) {
    const int64_t rows = n / inner_size;
    const int64_t inner_vecs = inner_size / 4;
    const int x_blocks = ClampBlocks(inner_vecs, kThreadsPerBlock);
    const int y_blocks = rows > kMaxGridY ? kMaxGridY : static_cast<int>(rows);
    const dim3 grid(x_blocks, y_blocks, 1);
    SelectRank1CondFloat4Kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        static_cast<const bool*>(cond), reinterpret_cast<const float4*>(then_t),
        reinterpret_cast<const float4*>(else_t), reinterpret_cast<float4*>(out),
        rows, inner_vecs);
    return;
  }
  LaunchSelectRank1CondTyped<float>(cond, then_t, else_t, out, n, inner_size,
                                    stream);
}

#define DEFINE_SELECT_LAUNCHERS(suffix, T)                                     \
  void LaunchMusaSelectSameShape##suffix(const void* cond, const void* then_t, \
                                         const void* else_t, void* out,        \
                                         int64_t n, musaStream_t stream) {     \
    LaunchSelectSameShapeTyped<T>(cond, then_t, else_t, out, n, stream);       \
  }                                                                            \
  void LaunchMusaSelectScalarCond##suffix(                                     \
      const void* cond, const void* then_t, const void* else_t, void* out,     \
      int64_t n, musaStream_t stream) {                                        \
    LaunchSelectScalarCondTyped<T>(cond, then_t, else_t, out, n, stream);      \
  }                                                                            \
  void LaunchMusaSelectRank1Cond##suffix(                                      \
      const void* cond, const void* then_t, const void* else_t, void* out,     \
      int64_t n, int64_t inner_size, musaStream_t stream) {                    \
    LaunchSelectRank1CondTyped<T>(cond, then_t, else_t, out, n, inner_size,    \
                                  stream);                                     \
  }

DEFINE_SELECT_LAUNCHERS(Half, half)
DEFINE_SELECT_LAUNCHERS(BFloat16, __mt_bfloat16)
DEFINE_SELECT_LAUNCHERS(Int32, int32_t)
DEFINE_SELECT_LAUNCHERS(Int64, int64_t)
DEFINE_SELECT_LAUNCHERS(Bool, bool)

#undef DEFINE_SELECT_LAUNCHERS

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
