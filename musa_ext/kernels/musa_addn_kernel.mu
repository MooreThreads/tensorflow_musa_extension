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

// 设备函数：加载不同数据类型的值
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

// 整数类型的支持
__device__ __forceinline__ int32_t LoadInt32(const int32_t* p) { return *p; }
__device__ __forceinline__ void StoreInt32(int32_t* p, int32_t v) { *p = v; }

// 使用 tensorflow::int64 而不是 int64_t
__device__ __forceinline__ tensorflow::int64 LoadInt64(const tensorflow::int64* p) { return *p; }
__device__ __forceinline__ void StoreInt64(tensorflow::int64* p, tensorflow::int64 v) { *p = v; }

// 双精度支持
__device__ __forceinline__ double LoadDouble(const double* p) { return *p; }
__device__ __forceinline__ void StoreDouble(double* p, double v) { *p = v; }

// 主要的AddN kernel模板 - 使用 const T* const* 参数类型
template <typename T>
__global__ void AddNKernel(const T* const* inputs, T* output, int num_inputs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 使用适当的数据类型进行累加
        T sum = inputs[0][idx];
        for (int i = 1; i < num_inputs; ++i) {
            sum += inputs[i][idx];
        }
        output[idx] = sum;
    }
}

// 特化版本：使用float中间计算（适用于半精度）
template <>
__global__ void AddNKernel<Eigen::half>(const Eigen::half* const* inputs, Eigen::half* output, int num_inputs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = LoadFloat(&inputs[0][idx]);
        for (int i = 1; i < num_inputs; ++i) {
            sum += LoadFloat(&inputs[i][idx]);
        }
        StoreFloat(&output[idx], sum);
    }
}

template <>
__global__ void AddNKernel<bfloat16>(const bfloat16* const* inputs, bfloat16* output, int num_inputs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = LoadFloat(&inputs[0][idx]);
        for (int i = 1; i < num_inputs; ++i) {
            sum += LoadFloat(&inputs[i][idx]);
        }
        StoreFloat(&output[idx], sum);
    }
}

// 启动函数 - 使用 const T* const* 参数类型
template <typename T>
void LaunchAddN(const T* const* inputs, T* output, int num_inputs, int n, musaStream_t stream) {
    if (n <= 0 || num_inputs <= 0) return;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    AddNKernel<T><<<blocks, threads, 0, stream>>>(inputs, output, num_inputs, n);
    
    // 检查kernel启动错误
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        // 错误处理将在C++层处理
    }
}

// 显式实例化 - 使用 tensorflow::int64
// LaunchAddN 是模板函数 必须有显示实例化 编译器才会在编译阶段生成函数代码
template void LaunchAddN<float>(const float* const*, float*, int, int, musaStream_t);
template void LaunchAddN<double>(const double* const*, double*, int, int, musaStream_t);
template void LaunchAddN<Eigen::half>(const Eigen::half* const*, Eigen::half*, int, int, musaStream_t);
template void LaunchAddN<bfloat16>(const bfloat16* const*, bfloat16*, int, int, musaStream_t);
template void LaunchAddN<int32_t>(const int32_t* const*, int32_t*, int, int, musaStream_t);
template void LaunchAddN<tensorflow::int64>(const tensorflow::int64* const*, tensorflow::int64*, int, int, musaStream_t);

} // namespace musa
} // namespace tensorflow