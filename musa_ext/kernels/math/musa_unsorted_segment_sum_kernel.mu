#include <musa_runtime.h>
#include <musa_fp16.h>
#include <stdint.h>

// Custom atomicAdd for int64_t (handles both long and long long)
__device__ __forceinline__ int64_t atomicAddInt64(int64_t* address, int64_t val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    (unsigned long long)((int64_t)assumed + val));
  } while (assumed != old);
  return (int64_t)old;
}

// Overloads for atomicAdd to handle int64_t
__device__ __forceinline__ long long atomicAdd(long long* address, long long val) {
  return (long long)atomicAddInt64((int64_t*)address, (int64_t)val);
}

// Handle toolchains where int64_t is an alias of long rather than long long.
#if (defined(__x86_64__) || defined(__aarch64__)) && !defined(__APPLE__)
__device__ __forceinline__ long atomicAdd(long* address, long val) {
  return (long)atomicAddInt64((int64_t*)address, (int64_t)val);
}
#endif

template <typename T, typename Tindex>
__global__ void UnsortedSegmentSumKernel(const T* data, const Tindex* segment_ids,
                                         Tindex num_segments, int64_t N, int64_t M,
                                         T* output) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N * M) {
    int64_t j = tid % M;
    int64_t i = tid / M;

    Tindex segment_id = segment_ids[i];

    if (segment_id >= 0 && segment_id < num_segments) {
      atomicAdd(&output[segment_id * M + j], data[tid]);
    }
  }
}

extern "C" {

#define DEFINE_SEGMENT_SUM_LAUNCHER(Name, T, Tindex) \
  void Name(const T* data, const Tindex* segment_ids, Tindex num_segments, \
            int64_t N, int64_t M, T* output, musaStream_t stream) { \
    int64_t total = N * M; \
    if (total == 0) return; \
    const int threads_per_block = 256; \
    const int blocks = (total + threads_per_block - 1) / threads_per_block; \
    UnsortedSegmentSumKernel<T, Tindex><<<blocks, threads_per_block, 0, stream>>>( \
        data, segment_ids, num_segments, N, M, output); \
  }

DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumFloatInt32, float, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumFloatInt64, float, int64_t)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumDoubleInt32, double, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumDoubleInt64, double, int64_t)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt32Int32, int, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt32Int64, int, int64_t)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt64Int32, int64_t, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt64Int64, int64_t, int64_t)

#undef DEFINE_SEGMENT_SUM_LAUNCHER

} // extern "C"

// ==========================================
// Half/BFloat16 support via float accumulation
// ==========================================
__device__ __forceinline__ float HalfBitsToFloat_(uint16_t bits) {
    return __half2float(*reinterpret_cast<const __half*>(&bits));
}
__device__ __forceinline__ uint16_t FloatToHalfBits_(float v) {
    __half h = __float2half(v);
    return *reinterpret_cast<uint16_t*>(&h);
}
__device__ __forceinline__ float BFloat16BitsToFloat_(uint16_t bits) {
    uint32_t full = (uint32_t)bits << 16;
    float result;
    __builtin_memcpy(&result, &full, sizeof(float));
    return result;
}
__device__ __forceinline__ uint16_t FloatToBFloat16Bits_(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

template <typename Tindex>
__global__ void UnsortedSegmentSumKernelHalfToFloat(
    const uint16_t* data, const Tindex* segment_ids, Tindex num_segments,
    int64_t N, int64_t M, float* float_output) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * M) {
        int64_t j = tid % M;
        int64_t i = tid / M;
        Tindex segment_id = segment_ids[i];
        if (segment_id >= 0 && segment_id < num_segments) {
            float val = HalfBitsToFloat_(data[tid]);
            atomicAdd(&float_output[segment_id * M + j], val);
        }
    }
}

template <typename Tindex>
__global__ void UnsortedSegmentSumKernelBFloat16ToFloat(
    const uint16_t* data, const Tindex* segment_ids, Tindex num_segments,
    int64_t N, int64_t M, float* float_output) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * M) {
        int64_t j = tid % M;
        int64_t i = tid / M;
        Tindex segment_id = segment_ids[i];
        if (segment_id >= 0 && segment_id < num_segments) {
            float val = BFloat16BitsToFloat_(data[tid]);
            atomicAdd(&float_output[segment_id * M + j], val);
        }
    }
}

__global__ void ConvertFloatToHalfBitsKernel(const float* input,
                                              uint16_t* output, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) output[tid] = FloatToHalfBits_(input[tid]);
}

__global__ void ConvertFloatToBFloat16BitsKernel(const float* input,
                                                  uint16_t* output, int64_t n) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) output[tid] = FloatToBFloat16Bits_(input[tid]);
}

extern "C" {
#define DEFINE_HALF_ACCUM_LAUNCHER(FuncName, AccumKernel, Tindex)               \
  void FuncName(const uint16_t* data, const Tindex* segment_ids,                \
                Tindex num_segments, int64_t N, int64_t M,                      \
                float* float_output, musaStream_t stream) {                     \
    int64_t total = N * M;                                                      \
    if (total == 0) return;                                                     \
    const int threads = 256;                                                    \
    const int blocks = static_cast<int>((total + threads - 1) / threads);      \
    AccumKernel<Tindex><<<blocks, threads, 0, stream>>>(                        \
        data, segment_ids, num_segments, N, M, float_output);                  \
  }

DEFINE_HALF_ACCUM_LAUNCHER(LaunchUnsortedSegmentSumHalfToFloatInt32,
                            UnsortedSegmentSumKernelHalfToFloat, int)
DEFINE_HALF_ACCUM_LAUNCHER(LaunchUnsortedSegmentSumHalfToFloatInt64,
                            UnsortedSegmentSumKernelHalfToFloat, int64_t)
DEFINE_HALF_ACCUM_LAUNCHER(LaunchUnsortedSegmentSumBFloat16ToFloatInt32,
                            UnsortedSegmentSumKernelBFloat16ToFloat, int)
DEFINE_HALF_ACCUM_LAUNCHER(LaunchUnsortedSegmentSumBFloat16ToFloatInt64,
                            UnsortedSegmentSumKernelBFloat16ToFloat, int64_t)

#undef DEFINE_HALF_ACCUM_LAUNCHER

void LaunchConvertFloatToHalfBits(const float* input, uint16_t* output,
                                   int64_t n, musaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    ConvertFloatToHalfBitsKernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

void LaunchConvertFloatToBFloat16Bits(const float* input, uint16_t* output,
                                       int64_t n, musaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    ConvertFloatToBFloat16BitsKernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

} // extern "C"
