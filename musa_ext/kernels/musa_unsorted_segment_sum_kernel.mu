#include <musa_runtime.h>
#include <stdint.h>

__device__ __forceinline__ long long atomicAdd(long long* address, long long val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    (unsigned long long)((long long)assumed + val));
  } while (assumed != old);
  return (long long)old;
}

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
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumFloatInt64, float, long long)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumDoubleInt32, double, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumDoubleInt64, double, long long)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt32Int32, int, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt32Int64, int, long long)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt64Int32, long long, int)
DEFINE_SEGMENT_SUM_LAUNCHER(LaunchUnsortedSegmentSumInt64Int64, long long, long long)

#undef DEFINE_SEGMENT_SUM_LAUNCHER

} // extern "C"