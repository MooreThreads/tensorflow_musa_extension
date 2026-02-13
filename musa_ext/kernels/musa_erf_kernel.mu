#include <musa_runtime.h>
#include <musa_fp16.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

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
    uint16_t* b_ptr = (uint16_t*)p;
    uint32_t* f_ptr = (uint32_t*)&res;
    *f_ptr = ((uint32_t)(*b_ptr)) << 16;
    return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
    uint32_t* f_ptr = (uint32_t*)&v;
    uint16_t b_val = (*f_ptr) >> 16;
    *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__global__ void ErfKernel(const T* src, T* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = LoadFloat(&src[i]);
        
        float res = erff(val);
        
        StoreFloat(&dst[i], res);
    }
}

template <>
__global__ void ErfKernel<double>(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = erf(src[i]);
    }
}

template <typename T>
void LaunchErf(const T* src, T* dst, int n, musaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ErfKernel<T><<<blocks, threads, 0, stream>>>(src, dst, n);
}

template void LaunchErf<float>(const float*, float*, int, musaStream_t);
template void LaunchErf<double>(const double*, double*, int, musaStream_t);
template void LaunchErf<Eigen::half>(const Eigen::half*, Eigen::half*, int, musaStream_t);
template void LaunchErf<bfloat16>(const bfloat16*, bfloat16*, int, musaStream_t);

} // namespace musa
} // namespace tensorflow

