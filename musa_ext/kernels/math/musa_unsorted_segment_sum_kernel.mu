#include <musa_runtime.h>
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

// Handle 'long' type when it's different from 'long long'
#if defined(__x86_64__) && !defined(__APPLE__)
// On Linux x86_64, long is 64-bit but distinct from long long in C++ type system
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