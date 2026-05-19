#include <musa_runtime.h>
#include <musa_fp16.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

using bfloat16 = tensorflow::bfloat16;

namespace {

constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ int64_t AtomicAddInt64(int64_t* address, int64_t val) {
  unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(
                        static_cast<int64_t>(assumed) + val));
  } while (assumed != old);
  return static_cast<int64_t>(old);
}

__device__ __forceinline__ void AtomicAddValue(float* address, float val) {
  atomicAdd(address, val);
}

__device__ __forceinline__ void AtomicAddValue(double* address, double val) {
  atomicAdd(address, val);
}

__device__ __forceinline__ void AtomicAddValue(int* address, int val) {
  atomicAdd(address, val);
}

__device__ __forceinline__ void AtomicAddValue(int64_t* address, int64_t val) {
  AtomicAddInt64(address, val);
}

__device__ __forceinline__ void AtomicAddValue(Eigen::half* address,
                                               Eigen::half val) {
  __half* h_addr = reinterpret_cast<__half*>(address);
  const float addend = __half2float(*reinterpret_cast<__half*>(&val));
  unsigned int* base = reinterpret_cast<unsigned int*>(
      reinterpret_cast<uintptr_t>(h_addr) & ~static_cast<uintptr_t>(0x3));
  const bool high = (reinterpret_cast<uintptr_t>(h_addr) & 0x2) != 0;
  unsigned int old = *base;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short old_bits = high ? static_cast<unsigned short>(old >> 16)
                                   : static_cast<unsigned short>(old & 0xffffu);
    __half old_half;
    *reinterpret_cast<unsigned short*>(&old_half) = old_bits;
    __half new_half = __float2half(__half2float(old_half) + addend);
    unsigned short new_bits = *reinterpret_cast<unsigned short*>(&new_half);
    unsigned int next = high ? ((old & 0x0000ffffu) |
                                (static_cast<unsigned int>(new_bits) << 16))
                             : ((old & 0xffff0000u) |
                                static_cast<unsigned int>(new_bits));
    old = atomicCAS(base, assumed, next);
  } while (assumed != old);
}

__device__ __forceinline__ float BFloat16ToFloat(unsigned short bits) {
  float value = 0.0f;
  unsigned int raw = static_cast<unsigned int>(bits) << 16;
  *reinterpret_cast<unsigned int*>(&value) = raw;
  return value;
}

__device__ __forceinline__ unsigned short FloatToBFloat16(float value) {
  unsigned int raw = *reinterpret_cast<unsigned int*>(&value);
  return static_cast<unsigned short>(raw >> 16);
}

__device__ __forceinline__ void AtomicAddValue(bfloat16* address, bfloat16 val) {
  unsigned short* bf_addr = reinterpret_cast<unsigned short*>(address);
  const float addend = BFloat16ToFloat(*reinterpret_cast<unsigned short*>(&val));
  unsigned int* base = reinterpret_cast<unsigned int*>(
      reinterpret_cast<uintptr_t>(bf_addr) & ~static_cast<uintptr_t>(0x3));
  const bool high = (reinterpret_cast<uintptr_t>(bf_addr) & 0x2) != 0;
  unsigned int old = *base;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short old_bits = high ? static_cast<unsigned short>(old >> 16)
                                   : static_cast<unsigned short>(old & 0xffffu);
    unsigned short new_bits = FloatToBFloat16(BFloat16ToFloat(old_bits) + addend);
    unsigned int next = high ? ((old & 0x0000ffffu) |
                                (static_cast<unsigned int>(new_bits) << 16))
                             : ((old & 0xffff0000u) |
                                static_cast<unsigned int>(new_bits));
    old = atomicCAS(base, assumed, next);
  } while (assumed != old);
}

__device__ __forceinline__ bool DecodeScatterOffset(
    const int* indices, int index_depth, int64_t update_idx, int64_t slice_idx,
    int64_t slice_size, const int64_t* strides, const int64_t* dim_sizes,
    int64_t output_elements, int64_t* output_offset) {
  int64_t offset = slice_idx;
  for (int d = 0; d < index_depth; ++d) {
    const int coord = indices[update_idx * index_depth + d];
    if (coord < 0 || static_cast<int64_t>(coord) >= dim_sizes[d]) return false;
    offset += static_cast<int64_t>(coord) * strides[d];
  }
  if (offset < 0 || offset >= output_elements) return false;
  *output_offset = offset;
  return true;
}

__device__ __forceinline__ bool DecodeScatterOffset(
    const int64_t* indices, int index_depth, int64_t update_idx, int64_t slice_idx,
    int64_t slice_size, const int64_t* strides, const int64_t* dim_sizes,
    int64_t output_elements, int64_t* output_offset) {
  int64_t offset = slice_idx;
  for (int d = 0; d < index_depth; ++d) {
    const int64_t coord = indices[update_idx * index_depth + d];
    if (coord < 0 || coord >= dim_sizes[d]) return false;
    offset += coord * strides[d];
  }
  if (offset < 0 || offset >= output_elements) return false;
  *output_offset = offset;
  return true;
}

template <typename T, typename IndexT>
__global__ void ScatterNdAddKernel(const IndexT* __restrict__ indices,
                                   const T* __restrict__ updates,
                                   T* __restrict__ output, int index_depth,
                                   int64_t num_updates, int64_t slice_size,
                                   const int64_t* __restrict__ strides,
                                   const int64_t* __restrict__ dim_sizes,
                                   int64_t output_elements) {
  const int64_t total = num_updates * slice_size;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; tid < total; tid += step) {
    const int64_t update_idx = tid / slice_size;
    const int64_t slice_idx = tid - update_idx * slice_size;
    int64_t output_offset = 0;
    if (DecodeScatterOffset(indices, index_depth, update_idx, slice_idx,
                            slice_size, strides, dim_sizes, output_elements,
                            &output_offset)) {
      AtomicAddValue(output + output_offset, updates[tid]);
    }
  }
}

template <typename T, typename IndexT>
void LaunchScatterNdAdd(const IndexT* indices, const T* updates, T* output,
                        int index_depth, int64_t num_updates, int64_t slice_size,
                        const int64_t* strides, const int64_t* dim_sizes,
                        int64_t output_elements, musaStream_t stream) {
  const int64_t total = num_updates * slice_size;
  if (total <= 0) return;
  int64_t blocks64 = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks64 > 4096) blocks64 = 4096;
  const int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
  ScatterNdAddKernel<T, IndexT><<<blocks, kThreadsPerBlock, 0, stream>>>(
      indices, updates, output, index_depth, num_updates, slice_size, strides,
      dim_sizes, output_elements);
}

template <typename T, typename IndexT>
__global__ void ResourceScatterAddRowsKernel(
    T* __restrict__ params, const IndexT* __restrict__ indices,
    const T* __restrict__ updates, int64_t num_updates, int64_t slice_size,
    IndexT limit) {
  const int64_t total = num_updates * slice_size;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; tid < total; tid += step) {
    const int64_t update_idx = tid / slice_size;
    const int64_t slice_idx = tid - update_idx * slice_size;
    const IndexT index = indices[update_idx];
    if (index < 0 || index >= limit) continue;
    AtomicAddValue(params + static_cast<int64_t>(index) * slice_size + slice_idx,
                   updates[tid]);
  }
}

template <typename T, typename IndexT>
void LaunchResourceScatterAddRows(T* params, const IndexT* indices,
                                  const T* updates, int64_t num_updates,
                                  int64_t slice_size, IndexT limit,
                                  musaStream_t stream) {
  const int64_t total = num_updates * slice_size;
  if (total <= 0) return;
  int64_t blocks64 = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks64 > 4096) blocks64 = 4096;
  const int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
  ResourceScatterAddRowsKernel<T, IndexT>
      <<<blocks, kThreadsPerBlock, 0, stream>>>(
          params, indices, updates, num_updates, slice_size, limit);
}

}  // namespace

extern "C" {

#define DEFINE_SCATTER_ND_ADD(Name, T, IndexT)                             \
  void Name(const IndexT* indices, const T* updates, T* output,             \
            int index_depth, int64_t num_updates, int64_t slice_size,       \
            const int64_t* strides, const int64_t* dim_sizes,               \
            int64_t output_elements, musaStream_t stream) {                \
    LaunchScatterNdAdd<T, IndexT>(indices, updates, output, index_depth,    \
                                  num_updates, slice_size, strides,         \
                                  dim_sizes, output_elements, stream);      \
  }

DEFINE_SCATTER_ND_ADD(LaunchScatterNdFloatInt32, float, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdFloatInt64, float, int64_t)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdDoubleInt32, double, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdDoubleInt64, double, int64_t)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdInt32Int32, int, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdInt32Int64, int, int64_t)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdInt64Int32, int64_t, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdInt64Int64, int64_t, int64_t)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdHalfInt32, Eigen::half, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdHalfInt64, Eigen::half, int64_t)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdBFloat16Int32, bfloat16, int)
DEFINE_SCATTER_ND_ADD(LaunchScatterNdBFloat16Int64, bfloat16, int64_t)

#undef DEFINE_SCATTER_ND_ADD

#define DEFINE_RESOURCE_SCATTER_ADD_ROWS(Name, T, IndexT)                 \
  void Name(T* params, const IndexT* indices, const T* updates,           \
            int64_t num_updates, int64_t slice_size, IndexT limit,        \
            musaStream_t stream) {                                       \
    LaunchResourceScatterAddRows<T, IndexT>(params, indices, updates,     \
                                            num_updates, slice_size,      \
                                            limit, stream);              \
  }

DEFINE_RESOURCE_SCATTER_ADD_ROWS(LaunchResourceScatterAddRowsFloatInt32,
                                 float, int)
DEFINE_RESOURCE_SCATTER_ADD_ROWS(LaunchResourceScatterAddRowsFloatInt64,
                                 float, int64_t)

#undef DEFINE_RESOURCE_SCATTER_ADD_ROWS

}  // extern "C"
