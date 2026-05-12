#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = 4;
constexpr int kMaxBlocks = 4096;
constexpr int kMaxGridY = 65535;

static inline int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

static inline int ClampBlocks(int64_t items, int64_t items_per_block) {
  int64_t blocks = CeilDiv(items, items_per_block);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : static_cast<int>(blocks);
}

static inline int ClampGridY(int64_t rows) {
  if (rows < 1) return 1;
  return rows > kMaxGridY ? kMaxGridY : static_cast<int>(rows);
}

static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs + rhs;
  }
};

template <typename T>
struct MulOp {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs * rhs;
  }
};

__global__ __launch_bounds__(kThreadsPerBlock) void AddFloat4ContiguousKernel(
    const float4* __restrict__ lhs, const float4* __restrict__ rhs,
    float4* __restrict__ out, int64_t vec_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    out[idx] = make_float4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulFloat4ContiguousKernel(
    const float4* __restrict__ lhs, const float4* __restrict__ rhs,
    float4* __restrict__ out, int64_t vec_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    out[idx] = make_float4(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void AddFloat4ScalarKernel(
    const float4* __restrict__ dense, const float* __restrict__ scalar,
    float4* __restrict__ out, int64_t vec_n) {
  const float s = scalar[0];
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 d = dense[idx];
    out[idx] = make_float4(d.x + s, d.y + s, d.z + s, d.w + s);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulFloat4ScalarKernel(
    const float4* __restrict__ dense, const float* __restrict__ scalar,
    float4* __restrict__ out, int64_t vec_n) {
  const float s = scalar[0];
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_n; idx += stride) {
    const float4 d = dense[idx];
    out[idx] = make_float4(d.x * s, d.y * s, d.z * s, d.w * s);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void AddTailVector2DFloatKernel(
    const float* __restrict__ dense, const float* __restrict__ tail_vector,
    float* __restrict__ out, int64_t rows, int64_t width, bool vector_on_left) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; col < width; col += stride) {
      const int64_t offset = row * width + col;
      const float d = dense[offset];
      const float v = tail_vector[col];
      out[offset] = vector_on_left ? v + d : d + v;
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) void MulTailVector2DFloatKernel(
    const float* __restrict__ dense, const float* __restrict__ tail_vector,
    float* __restrict__ out, int64_t rows, int64_t width, bool vector_on_left) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; col < width; col += stride) {
      const int64_t offset = row * width + col;
      const float d = dense[offset];
      const float v = tail_vector[col];
      out[offset] = vector_on_left ? v * d : d * v;
    }
  }
}

struct BFloat16AddOp {
  __device__ __forceinline__ __mt_bfloat16 operator()(__mt_bfloat16 lhs,
                                                      __mt_bfloat16 rhs) const {
    return __float2bfloat16(__bfloat162float(lhs) + __bfloat162float(rhs));
  }
};

struct BFloat16MulOp {
  __device__ __forceinline__ __mt_bfloat16 operator()(__mt_bfloat16 lhs,
                                                      __mt_bfloat16 rhs) const {
    return __float2bfloat16(__bfloat162float(lhs) * __bfloat162float(rhs));
  }
};

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryContiguousKernel(
    const T* __restrict__ lhs, const T* __restrict__ rhs, T* __restrict__ out,
    int64_t n, Op op) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = op(lhs[offset], rhs[offset]);
      }
    }
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryScalarKernel(
    const T* __restrict__ dense, const T* __restrict__ scalar,
    T* __restrict__ out, int64_t n, Op op, bool scalar_on_left) {
  const T scalar_value = scalar[0];
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        out[offset] = scalar_on_left ? op(scalar_value, dense[offset])
                                     : op(dense[offset], scalar_value);
      }
    }
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(kThreadsPerBlock) void BinaryTailVectorKernel(
    const T* __restrict__ dense, const T* __restrict__ tail_vector,
    T* __restrict__ out, int64_t n, int64_t width, Op op, bool vector_on_left) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                kItemsPerThread;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * kItemsPerThread;
  for (; idx < n; idx += stride) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int64_t offset = idx + i;
      if (offset < n) {
        const T vector_value = tail_vector[offset % width];
        out[offset] = vector_on_left ? op(vector_value, dense[offset])
                                     : op(dense[offset], vector_value);
      }
    }
  }
}

template <typename T, typename Op>
void LaunchContiguousTyped(const T* lhs, const T* rhs, T* out, int64_t n,
                           musaStream_t stream, Op op) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryContiguousKernel<T, Op>
      <<<blocks, kThreadsPerBlock, 0, stream>>>(lhs, rhs, out, n, op);
}

template <typename T, typename Op>
void LaunchScalarTyped(const T* dense, const T* scalar, T* out, int64_t n,
                       bool scalar_on_left, musaStream_t stream, Op op) {
  if (n <= 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryScalarKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, scalar, out, n, op, scalar_on_left);
}

template <typename T, typename Op>
void LaunchTailVectorTyped(const T* dense, const T* tail_vector, T* out,
                           int64_t n, int64_t width, bool vector_on_left,
                           musaStream_t stream, Op op) {
  if (n <= 0 || width <= 0 || n % width != 0) return;
  const int blocks = ClampBlocks(n, kThreadsPerBlock * kItemsPerThread);
  BinaryTailVectorKernel<T, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, tail_vector, out, n, width, op, vector_on_left);
}

}  // namespace

extern "C" {

void LaunchMusaBinaryAddContiguousFloat(const float* lhs, const float* rhs,
                                        float* out, int64_t n,
                                        musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(lhs) && IsAligned16(rhs) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    AddFloat4ContiguousKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<float>());
}

void LaunchMusaBinaryAddScalarFloat(const float* dense, const float* scalar,
                                    float* out, int64_t n, bool scalar_on_left,
                                    musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(dense) && IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    AddFloat4ScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar,
        reinterpret_cast<float4*>(out), vec_n);
    return;
  }
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<float>());
}

void LaunchMusaBinaryAddTailVectorFloat(const float* dense,
                                        const float* tail_vector, float* out,
                                        int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  if (n > 0 && width > 0 && n % width == 0 && width <= 4096) {
    const int64_t rows = n / width;
    const int x_blocks = ClampBlocks(width, kThreadsPerBlock);
    const dim3 grid(x_blocks, ClampGridY(rows), 1);
    AddTailVector2DFloatKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, rows, width, vector_on_left);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<float>());
}

void LaunchMusaBinaryMulContiguousFloat(const float* lhs, const float* rhs,
                                        float* out, int64_t n,
                                        musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(lhs) && IsAligned16(rhs) &&
      IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    MulFloat4ContiguousKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs), reinterpret_cast<float4*>(out),
        vec_n);
    return;
  }
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<float>());
}

void LaunchMusaBinaryMulScalarFloat(const float* dense, const float* scalar,
                                    float* out, int64_t n, bool scalar_on_left,
                                    musaStream_t stream) {
  if (n >= 4 && (n % 4) == 0 && IsAligned16(dense) && IsAligned16(out)) {
    const int64_t vec_n = n / 4;
    const int blocks = ClampBlocks(vec_n, kThreadsPerBlock);
    MulFloat4ScalarKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar,
        reinterpret_cast<float4*>(out), vec_n);
    return;
  }
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<float>());
}

void LaunchMusaBinaryMulTailVectorFloat(const float* dense,
                                        const float* tail_vector, float* out,
                                        int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  if (n > 0 && width > 0 && n % width == 0 && width <= 4096) {
    const int64_t rows = n / width;
    const int x_blocks = ClampBlocks(width, kThreadsPerBlock);
    const dim3 grid(x_blocks, ClampGridY(rows), 1);
    MulTailVector2DFloatKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        dense, tail_vector, out, rows, width, vector_on_left);
    return;
  }
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<float>());
}

void LaunchMusaBinaryAddContiguousHalf(const half* lhs, const half* rhs,
                                       half* out, int64_t n,
                                       musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<half>());
}

void LaunchMusaBinaryAddScalarHalf(const half* dense, const half* scalar,
                                   half* out, int64_t n, bool scalar_on_left,
                                   musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<half>());
}

void LaunchMusaBinaryAddTailVectorHalf(const half* dense,
                                       const half* tail_vector, half* out,
                                       int64_t n, int64_t width,
                                       bool vector_on_left,
                                       musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<half>());
}

void LaunchMusaBinaryMulContiguousHalf(const half* lhs, const half* rhs,
                                       half* out, int64_t n,
                                       musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<half>());
}

void LaunchMusaBinaryMulScalarHalf(const half* dense, const half* scalar,
                                   half* out, int64_t n, bool scalar_on_left,
                                   musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<half>());
}

void LaunchMusaBinaryMulTailVectorHalf(const half* dense,
                                       const half* tail_vector, half* out,
                                       int64_t n, int64_t width,
                                       bool vector_on_left,
                                       musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<half>());
}

void LaunchMusaBinaryAddContiguousBFloat16(const __mt_bfloat16* lhs,
                                           const __mt_bfloat16* rhs,
                                           __mt_bfloat16* out, int64_t n,
                                           musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, BFloat16AddOp());
}

void LaunchMusaBinaryAddScalarBFloat16(const __mt_bfloat16* dense,
                                       const __mt_bfloat16* scalar,
                                       __mt_bfloat16* out, int64_t n,
                                       bool scalar_on_left,
                                       musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    BFloat16AddOp());
}

void LaunchMusaBinaryAddTailVectorBFloat16(const __mt_bfloat16* dense,
                                           const __mt_bfloat16* tail_vector,
                                           __mt_bfloat16* out, int64_t n,
                                           int64_t width, bool vector_on_left,
                                           musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, BFloat16AddOp());
}

void LaunchMusaBinaryMulContiguousBFloat16(const __mt_bfloat16* lhs,
                                           const __mt_bfloat16* rhs,
                                           __mt_bfloat16* out, int64_t n,
                                           musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, BFloat16MulOp());
}

void LaunchMusaBinaryMulScalarBFloat16(const __mt_bfloat16* dense,
                                       const __mt_bfloat16* scalar,
                                       __mt_bfloat16* out, int64_t n,
                                       bool scalar_on_left,
                                       musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    BFloat16MulOp());
}

void LaunchMusaBinaryMulTailVectorBFloat16(const __mt_bfloat16* dense,
                                           const __mt_bfloat16* tail_vector,
                                           __mt_bfloat16* out, int64_t n,
                                           int64_t width, bool vector_on_left,
                                           musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, BFloat16MulOp());
}

void LaunchMusaBinaryAddContiguousInt32(const int32_t* lhs, const int32_t* rhs,
                                        int32_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<int32_t>());
}

void LaunchMusaBinaryAddScalarInt32(const int32_t* dense, const int32_t* scalar,
                                    int32_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<int32_t>());
}

void LaunchMusaBinaryAddTailVectorInt32(const int32_t* dense,
                                        const int32_t* tail_vector,
                                        int32_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<int32_t>());
}

void LaunchMusaBinaryMulContiguousInt32(const int32_t* lhs, const int32_t* rhs,
                                        int32_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<int32_t>());
}

void LaunchMusaBinaryMulScalarInt32(const int32_t* dense, const int32_t* scalar,
                                    int32_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<int32_t>());
}

void LaunchMusaBinaryMulTailVectorInt32(const int32_t* dense,
                                        const int32_t* tail_vector,
                                        int32_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<int32_t>());
}

void LaunchMusaBinaryAddContiguousInt64(const int64_t* lhs, const int64_t* rhs,
                                        int64_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, AddOp<int64_t>());
}

void LaunchMusaBinaryAddScalarInt64(const int64_t* dense, const int64_t* scalar,
                                    int64_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    AddOp<int64_t>());
}

void LaunchMusaBinaryAddTailVectorInt64(const int64_t* dense,
                                        const int64_t* tail_vector,
                                        int64_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, AddOp<int64_t>());
}

void LaunchMusaBinaryMulContiguousInt64(const int64_t* lhs, const int64_t* rhs,
                                        int64_t* out, int64_t n,
                                        musaStream_t stream) {
  LaunchContiguousTyped(lhs, rhs, out, n, stream, MulOp<int64_t>());
}

void LaunchMusaBinaryMulScalarInt64(const int64_t* dense, const int64_t* scalar,
                                    int64_t* out, int64_t n,
                                    bool scalar_on_left, musaStream_t stream) {
  LaunchScalarTyped(dense, scalar, out, n, scalar_on_left, stream,
                    MulOp<int64_t>());
}

void LaunchMusaBinaryMulTailVectorInt64(const int64_t* dense,
                                        const int64_t* tail_vector,
                                        int64_t* out, int64_t n, int64_t width,
                                        bool vector_on_left,
                                        musaStream_t stream) {
  LaunchTailVectorTyped(dense, tail_vector, out, n, width, vector_on_left,
                        stream, MulOp<int64_t>());
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
