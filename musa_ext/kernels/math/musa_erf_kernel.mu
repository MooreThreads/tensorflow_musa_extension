#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>
#include <musa_fp16.h>
#if defined(__has_include)
#if __has_include(<musa_bf16.h>)
#include <musa_bf16.h>
#define TF_MUSA_HAS_BFLOAT16_INTRINSICS 1
#else
#define TF_MUSA_HAS_BFLOAT16_INTRINSICS 0
#endif
#else
#define TF_MUSA_HAS_BFLOAT16_INTRINSICS 0
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kScalarILP = 4;
constexpr int kPairILP = 2;
constexpr int kMaxBlocks = 4096;

__host__ __device__ __forceinline__ int DivUp(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __forceinline__ int ClampBlocks(int items, int items_per_block) {
    int blocks = DivUp(items, items_per_block);
    if (blocks < 1) {
        return 1;
    }
    return blocks > kMaxBlocks ? kMaxBlocks : blocks;
}

template <typename VecT, typename T>
__host__ __forceinline__ bool IsAlignedForVec(const T* ptr) {
    const uintptr_t alignment =
        alignof(VecT) > sizeof(VecT) ? alignof(VecT) : sizeof(VecT);
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

}  // namespace

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
    const __half* h_ptr = reinterpret_cast<const __half*>(p);
    return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
    __half h = __float2half(v);
    *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
    float res = 0.0f;
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
    uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
    *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
    return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
    const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&v);
    uint16_t b_val = static_cast<uint16_t>((*f_ptr) >> 16);
    *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T, int kILP>
__global__ __launch_bounds__(kThreadsPerBlock) void ErfKernel(
    const T* __restrict__ src, T* __restrict__ dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * kILP;
    const int stride = blockDim.x * gridDim.x * kILP;

    for (; i < n; i += stride) {
#pragma unroll
        for (int j = 0; j < kILP; ++j) {
            const int idx = i + j;
            if (idx < n) {
                const float val = LoadFloat(src + idx);
                StoreFloat(dst + idx, erff(val));
            }
        }
    }
}

template <typename T, int kILP>
__global__ __launch_bounds__(kThreadsPerBlock) void ErfSpecialCaseFixupKernel(
    const T* __restrict__ src, T* __restrict__ dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * kILP;
    const int stride = blockDim.x * gridDim.x * kILP;

    for (; i < n; i += stride) {
#pragma unroll
        for (int j = 0; j < kILP; ++j) {
            const int idx = i + j;
            if (idx < n) {
                const float val = LoadFloat(src + idx);
                if (isnan(val)) {
                    StoreFloat(dst + idx, val);
                } else if (isinf(val)) {
                    StoreFloat(dst + idx, val > 0.0f ? 1.0f : -1.0f);
                }
            }
        }
    }
}

template <int kILP>
__global__ __launch_bounds__(kThreadsPerBlock) void ErfKernelDouble(
    const double* __restrict__ src, double* __restrict__ dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * kILP;
    const int stride = blockDim.x * gridDim.x * kILP;

    for (; i < n; i += stride) {
#pragma unroll
        for (int j = 0; j < kILP; ++j) {
            const int idx = i + j;
            if (idx < n) {
                dst[idx] = erf(src[idx]);
            }
        }
    }
}

__global__ __launch_bounds__(kThreadsPerBlock) void ErfHalf2Kernel(
    const Eigen::half* __restrict__ src, Eigen::half* __restrict__ dst,
    int pair_count, int n) {
    const __half2* src_vec = reinterpret_cast<const __half2*>(src);
    __half2* dst_vec = reinterpret_cast<__half2*>(dst);

    int pair_idx = (blockIdx.x * blockDim.x + threadIdx.x) * kPairILP;
    const int stride = blockDim.x * gridDim.x * kPairILP;

    for (; pair_idx < pair_count; pair_idx += stride) {
#pragma unroll
        for (int j = 0; j < kPairILP; ++j) {
            const int idx = pair_idx + j;
            if (idx < pair_count) {
                float2 vals = __half22float2(src_vec[idx]);
                vals.x = erff(vals.x);
                vals.y = erff(vals.y);
                dst_vec[idx] = __float22half2_rn(vals);
            }
        }
    }

    if ((n & 1) && blockIdx.x == 0 && threadIdx.x == 0) {
        const int tail_idx = pair_count * 2;
        StoreFloat(dst + tail_idx, erff(LoadFloat(src + tail_idx)));
    }
}

#if TF_MUSA_HAS_BFLOAT16_INTRINSICS
__global__ __launch_bounds__(kThreadsPerBlock) void ErfBFloat162Kernel(
    const bfloat16* __restrict__ src, bfloat16* __restrict__ dst,
    int pair_count, int n) {
    const __mt_bfloat162* src_vec =
        reinterpret_cast<const __mt_bfloat162*>(src);
    __mt_bfloat162* dst_vec = reinterpret_cast<__mt_bfloat162*>(dst);

    int pair_idx = (blockIdx.x * blockDim.x + threadIdx.x) * kPairILP;
    const int stride = blockDim.x * gridDim.x * kPairILP;

    for (; pair_idx < pair_count; pair_idx += stride) {
#pragma unroll
        for (int j = 0; j < kPairILP; ++j) {
            const int idx = pair_idx + j;
            if (idx < pair_count) {
                const __mt_bfloat162 packed = src_vec[idx];
                float2 vals = make_float2(__low2float(packed),
                                          __high2float(packed));
                vals.x = erff(vals.x);
                vals.y = erff(vals.y);
                dst_vec[idx] = __floats2bfloat162_rn(vals.x, vals.y);
            }
        }
    }

    if ((n & 1) && blockIdx.x == 0 && threadIdx.x == 0) {
        const int tail_idx = pair_count * 2;
        StoreFloat(dst + tail_idx, erff(LoadFloat(src + tail_idx)));
    }
}
#endif

template <typename T>
void LaunchScalarErf(const T* src, T* dst, int n, musaStream_t stream) {
    const int blocks = ClampBlocks(n, kThreadsPerBlock * kScalarILP);
    ErfKernel<T, kScalarILP><<<blocks, kThreadsPerBlock, 0, stream>>>(src, dst, n);
}

template <typename T>
void LaunchErf(const T* src, T* dst, int n, musaStream_t stream) {
    if (n <= 0) return;
    LaunchScalarErf(src, dst, n, stream);
}

template <typename T>
void LaunchErfSpecialCaseFixup(const T* src, T* dst, int n,
                               musaStream_t stream) {
    if (n <= 0) return;
    const int blocks = ClampBlocks(n, kThreadsPerBlock * kScalarILP);
    ErfSpecialCaseFixupKernel<T, kScalarILP>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(src, dst, n);
}

template <>
void LaunchErf<double>(const double* src, double* dst, int n,
                       musaStream_t stream) {
    if (n <= 0) return;
    const int blocks = ClampBlocks(n, kThreadsPerBlock * kScalarILP);
    ErfKernelDouble<kScalarILP><<<blocks, kThreadsPerBlock, 0, stream>>>(
        src, dst, n);
}

template <>
void LaunchErf<Eigen::half>(const Eigen::half* src, Eigen::half* dst, int n,
                            musaStream_t stream) {
    if (n <= 0) return;

    if (n >= 2 && IsAlignedForVec<__half2>(src) && IsAlignedForVec<__half2>(dst)) {
        const int pair_count = n / 2;
        const int blocks = ClampBlocks(pair_count, kThreadsPerBlock * kPairILP);
        ErfHalf2Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            src, dst, pair_count, n);
        return;
    }

    LaunchScalarErf(src, dst, n, stream);
}

template <>
void LaunchErf<bfloat16>(const bfloat16* src, bfloat16* dst, int n,
                         musaStream_t stream) {
    if (n <= 0) return;

#if TF_MUSA_HAS_BFLOAT16_INTRINSICS
    if (n >= 2 && IsAlignedForVec<__mt_bfloat162>(src) &&
        IsAlignedForVec<__mt_bfloat162>(dst)) {
        const int pair_count = n / 2;
        const int blocks = ClampBlocks(pair_count, kThreadsPerBlock * kPairILP);
        ErfBFloat162Kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            src, dst, pair_count, n);
        return;
    }
#endif

    LaunchScalarErf(src, dst, n, stream);
}

template void LaunchErf<float>(const float*, float*, int, musaStream_t);
template void LaunchErfSpecialCaseFixup<float>(const float*, float*, int,
                                               musaStream_t);
template void LaunchErfSpecialCaseFixup<Eigen::half>(const Eigen::half*,
                                                     Eigen::half*, int,
                                                     musaStream_t);
template void LaunchErfSpecialCaseFixup<bfloat16>(const bfloat16*, bfloat16*,
                                                  int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
