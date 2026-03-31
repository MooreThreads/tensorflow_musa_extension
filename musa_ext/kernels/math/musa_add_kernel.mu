#include <musa_runtime.h>

#include <stdint.h>

namespace tensorflow {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;

static inline int64_t CeilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

}  // namespace

extern "C" {

__global__ void AddContiguousKernelFloat(const float* lhs, const float* rhs,
                                         float* output, int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = lhs[idx] + rhs[idx];
  }
}

__global__ void AddContiguousKernelFloat4(const float4* lhs, const float4* rhs,
                                          float4* output, int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    float4 out;
    out.x = l.x + r.x;
    out.y = l.y + r.y;
    out.z = l.z + r.z;
    out.w = l.w + r.w;
    output[idx] = out;
  }
}

__global__ void AddScalarKernelFloat(const float* dense, const float scalar,
                                     float* output, int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = dense[idx] + scalar;
  }
}

__global__ void AddScalarKernelFloat4(const float4* dense, const float4 scalar4,
                                      float4* output, int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float4 d = dense[idx];
    float4 out;
    out.x = d.x + scalar4.x;
    out.y = d.y + scalar4.y;
    out.z = d.z + scalar4.z;
    out.w = d.w + scalar4.w;
    output[idx] = out;
  }
}

__global__ void AddTailVectorKernelFloat(const float* dense,
                                         const float* tail_vector,
                                         float* output, int64_t size,
                                         int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    output[idx] = dense[idx] + tail_vector[col];
  }
}

void LaunchMusaAddContiguousFloat(const float* lhs, const float* rhs,
                                  float* output, int64_t size,
                                  musaStream_t stream) {
  if (size <= 0) {
    return;
  }

  if (size >= 4 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(output)) {
    const int64_t vec_size = size / 4;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddContiguousKernelFloat4<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs),
        reinterpret_cast<float4*>(output), vec_size);

    const int64_t tail = size - vec_size * 4;
    if (tail > 0) {
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddContiguousKernelFloat<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          lhs + vec_size * 4, rhs + vec_size * 4, output + vec_size * 4, tail);
    }
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddContiguousKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      lhs, rhs, output, size);
}

void LaunchMusaAddScalarFloat(const float* dense, const float* scalar,
                              float* output, int64_t size,
                              musaStream_t stream) {
  if (size <= 0) {
    return;
  }
  const float scalar_value = scalar[0];

  if (size >= 4 && IsAligned16(dense) && IsAligned16(output)) {
    float4 scalar4;
    scalar4.x = scalar_value;
    scalar4.y = scalar_value;
    scalar4.z = scalar_value;
    scalar4.w = scalar_value;

    const int64_t vec_size = size / 4;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddScalarKernelFloat4<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar4,
        reinterpret_cast<float4*>(output), vec_size);

    const int64_t tail = size - vec_size * 4;
    if (tail > 0) {
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddScalarKernelFloat<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          dense + vec_size * 4, scalar_value, output + vec_size * 4, tail);
    }
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddScalarKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, scalar_value, output, size);
}

void LaunchMusaAddTailVectorFloat(const float* dense, const float* tail_vector,
                                  float* output, int64_t size, int64_t width,
                                  musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) {
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddTailVectorKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, tail_vector, output, size, width);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
