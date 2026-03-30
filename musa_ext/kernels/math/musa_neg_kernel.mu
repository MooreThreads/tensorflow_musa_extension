#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "tensorflow/core/framework/bfloat16.h"

#pragma GCC diagnostic pop

using bfloat16 = tensorflow::bfloat16;

namespace tensorflow {
namespace musa {

// Device function for negation with optimized half-precision implementations
template <typename T>
__device__ __forceinline__ T DeviceNeg(T val) {
  return -val;
}

// Specialized half negation using sign bit flip (faster than arithmetic)
template <>
__device__ __forceinline__ Eigen::half DeviceNeg<Eigen::half>(Eigen::half val) {
  uint16_t raw = *reinterpret_cast<const uint16_t*>(&val);
  raw ^= 0x8000;
  return *reinterpret_cast<const Eigen::half*>(&raw);
}

// Specialized bfloat16 negation using sign bit flip
template <>
__device__ __forceinline__ bfloat16 DeviceNeg<bfloat16>(bfloat16 val) {
  uint16_t raw = *reinterpret_cast<const uint16_t*>(&val);
  raw ^= 0x8000;
  return *reinterpret_cast<const bfloat16*>(&raw);
}

// Vector type loader for memory coalescing optimization
template <typename T, int VEC_SIZE>
struct VectorLoader;

// float specialization
template <>
struct VectorLoader<float, 4> {
  using VecType = float4;
  static __device__ __forceinline__ float4 load(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
  }
  static __device__ __forceinline__ void store(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
  }
  static __device__ __forceinline__ float4 negate(float4 val) {
    return make_float4(-val.x, -val.y, -val.z, -val.w);
  }
};

// double specialization
template <>
struct VectorLoader<double, 2> {
  using VecType = double2;
  static __device__ __forceinline__ double2 load(const double* ptr) {
    return *reinterpret_cast<const double2*>(ptr);
  }
  static __device__ __forceinline__ void store(double* ptr, double2 val) {
    *reinterpret_cast<double2*>(ptr) = val;
  }
  static __device__ __forceinline__ double2 negate(double2 val) {
    return make_double2(-val.x, -val.y);
  }
};

// int specialization
template <>
struct VectorLoader<int, 4> {
  using VecType = ::int4;
  static __device__ __forceinline__ ::int4 load(const int* ptr) {
    return *reinterpret_cast<const ::int4*>(ptr);
  }
  static __device__ __forceinline__ void store(int* ptr, ::int4 val) {
    *reinterpret_cast<::int4*>(ptr) = val;
  }
  static __device__ __forceinline__ ::int4 negate(::int4 val) {
    return make_int4(-val.x, -val.y, -val.z, -val.w);
  }
};

// int64_t specialization
template <>
struct VectorLoader<int64_t, 2> {
  using VecType = longlong2;
  static __device__ __forceinline__ longlong2 load(const int64_t* ptr) {
    return *reinterpret_cast<const longlong2*>(ptr);
  }
  static __device__ __forceinline__ void store(int64_t* ptr, longlong2 val) {
    *reinterpret_cast<longlong2*>(ptr) = val;
  }
  static __device__ __forceinline__ longlong2 negate(longlong2 val) {
    return make_longlong2(-val.x, -val.y);
  }
};

// half specialization (process 8 elements at once for better memory bandwidth)
template <>
struct VectorLoader<Eigen::half, 8> {
  using VecType = ::uint4;  // 8 half = 16 bytes
  static __device__ __forceinline__ ::uint4 load(const Eigen::half* ptr) {
    return *reinterpret_cast<const ::uint4*>(ptr);
  }
  static __device__ __forceinline__ void store(Eigen::half* ptr, ::uint4 val) {
    *reinterpret_cast<::uint4*>(ptr) = val;
  }
  static __device__ __forceinline__ ::uint4 negate(::uint4 val) {
    // Flip sign bits for all 8 half values at once
    // Each half has its sign bit at bit 15
    val.x ^= 0x80008000u;
    val.y ^= 0x80008000u;
    val.z ^= 0x80008000u;
    val.w ^= 0x80008000u;
    return val;
  }
};

// bfloat16 specialization (process 8 elements at once)
template <>
struct VectorLoader<bfloat16, 8> {
  using VecType = ::uint4;  // 8 bfloat16 = 16 bytes
  static __device__ __forceinline__ ::uint4 load(const bfloat16* ptr) {
    return *reinterpret_cast<const ::uint4*>(ptr);
  }
  static __device__ __forceinline__ void store(bfloat16* ptr, ::uint4 val) {
    *reinterpret_cast<::uint4*>(ptr) = val;
  }
  static __device__ __forceinline__ ::uint4 negate(::uint4 val) {
    // Flip sign bits for all 8 bfloat16 values at once
    val.x ^= 0x80008000u;
    val.y ^= 0x80008000u;
    val.z ^= 0x80008000u;
    val.w ^= 0x80008000u;
    return val;
  }
};

// Optimized scalar kernel for small sizes or remainder elements
template <typename T>
__global__ void NegKernelScalar(const T* __restrict__ in, T* __restrict__ out, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = DeviceNeg(in[idx]);
  }
}

// Unified optimized kernel that handles in-place operation efficiently
template <typename T, int VEC_SIZE>
__global__ void NegKernelOptimized(const T* __restrict__ in, T* __restrict__ out,
                                    int vec_count, int total_size, bool in_place) {
  using Loader = VectorLoader<T, VEC_SIZE>;
  using VecType = typename Loader::VecType;

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // For in-place operations, we can safely process in place since negation
  // is element-wise independent and doesn't depend on neighboring values
  if (!in_place || in != out) {
    // Process vectorized elements with grid-stride loop
    for (int i = tid; i < vec_count; i += stride) {
      VecType val = Loader::load(in + i * VEC_SIZE);
      val = Loader::negate(val);
      Loader::store(out + i * VEC_SIZE, val);
    }

    // Handle remainder elements
    const int remainder_start = vec_count * VEC_SIZE;
    for (int i = remainder_start + tid; i < total_size; i += stride) {
      out[i] = DeviceNeg(in[i]);
    }
  } else {
    // In-place optimization: same code but compiler knows pointers alias
    T* data = out;
    for (int i = tid; i < vec_count; i += stride) {
      VecType val = Loader::load(data + i * VEC_SIZE);
      val = Loader::negate(val);
      Loader::store(data + i * VEC_SIZE, val);
    }

    const int remainder_start = vec_count * VEC_SIZE;
    for (int i = remainder_start + tid; i < total_size; i += stride) {
      data[i] = DeviceNeg(data[i]);
    }
  }
}

// Configuration for optimal launch parameters
constexpr int kBlockSize = 256;
constexpr int kMaxGridSize = 65536;

template <typename T>
struct VectorConfig {
  static constexpr int vec_size = 1;
};

template<>
struct VectorConfig<float> {
  static constexpr int vec_size = 4;
};

template<>
struct VectorConfig<double> {
  static constexpr int vec_size = 2;
};

template<>
struct VectorConfig<int> {
  static constexpr int vec_size = 4;
};

template<>
struct VectorConfig<int64_t> {
  static constexpr int vec_size = 2;
};

template<>
struct VectorConfig<Eigen::half> {
  static constexpr int vec_size = 8;
};

template<>
struct VectorConfig<bfloat16> {
  static constexpr int vec_size = 8;
};

template <typename T>
void MusaNegKernelLauncher(const void* in, void* out, int size, musaStream_t stream) {
  if (size == 0) return;

  constexpr int vec_size = VectorConfig<T>::vec_size;
  const int vec_count = size / vec_size;
  const bool in_place = (in == out);

  // Choose optimal grid size
  int grid_size = (size + kBlockSize * vec_size - 1) / (kBlockSize * vec_size);
  grid_size = min(grid_size, kMaxGridSize);

  if (vec_count > 0) {
    // Use vectorized kernel
    NegKernelOptimized<T, vec_size><<<grid_size, kBlockSize, 0, stream>>>(
        static_cast<const T*>(in), static_cast<T*>(out),
        vec_count, size, in_place);
  } else {
    // Fall back to scalar kernel for very small sizes
    int scalar_grid = (size + kBlockSize - 1) / kBlockSize;
    NegKernelScalar<T><<<scalar_grid, kBlockSize, 0, stream>>>(
        static_cast<const T*>(in), static_cast<T*>(out), size);
  }
}

template void MusaNegKernelLauncher<float>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<double>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<int>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<int64_t>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<Eigen::half>(const void*, void*, int, musaStream_t);
template void MusaNegKernelLauncher<bfloat16>(const void*, void*, int, musaStream_t);

} // namespace musa
} // namespace tensorflow
