#include <musa_runtime.h>
#include <cstdint>
#include <cmath>

// ==========================================
// 1. Device 端工具函数: Philox4x32 RNG
// ==========================================

// Philox 需要 128-bit 的状态 (Counter) 和 64-bit 的 Key
struct uint4_ { uint32_t x, y, z, w; };
struct uint2_ { uint32_t x, y; };

__device__ __forceinline__ void philox4x32_bump(uint4_* counter) {
    if (++counter->x == 0) {
        if (++counter->y == 0) {
            if (++counter->z == 0) {
                ++counter->w;
            }
        }
    }
}

__device__ __forceinline__ uint4_ philox4x32_round(uint4_ counter, uint2_ key) {
    uint4_ result = counter;
    const uint32_t kPhiloxM4x32_0 = 0xD2511F53;
    const uint32_t kPhiloxM4x32_1 = 0xCD9E8D57;
    const uint32_t kPhiloxW32_0 = 0x9E3779B9;
    const uint32_t kPhiloxW32_1 = 0xBB67AE85;

    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        uint32_t lo0 = result.x * kPhiloxM4x32_0;
        uint32_t hi0 = __umulhi(result.x, kPhiloxM4x32_0);
        uint32_t lo1 = result.z * kPhiloxM4x32_1;
        uint32_t hi1 = __umulhi(result.z, kPhiloxM4x32_1);

        result.x = hi1 ^ result.y ^ key.y;
        result.y = lo1;
        result.z = hi0 ^ result.w ^ key.x;
        result.w = lo0;

        key.x += kPhiloxW32_0;
        key.y += kPhiloxW32_1;
    }
    return result;
}

// 类型转换辅助函数
template <typename T>
__device__ __forceinline__ T Uint32ToReal(uint32_t x);

template <>
__device__ __forceinline__ float Uint32ToReal<float>(uint32_t x) {
    // [0, 1) float
    return (x & 0x7FFFFF) * 1.192092896e-07F; 
}

template <>
__device__ __forceinline__ double Uint32ToReal<double>(uint32_t x) {
    // [0, 1) double
    return x * 2.3283064365386963e-10; 
}

template <typename T>
__device__ __forceinline__ void BoxMuller(uint32_t u1_raw, uint32_t u2_raw, T* z1, T* z2) {
    T u1 = Uint32ToReal<T>(u1_raw) + static_cast<T>(1e-20);
    T u2 = Uint32ToReal<T>(u2_raw);
    const T two_pi = static_cast<T>(6.283185307179586476925);
    
    T r = sqrt(static_cast<T>(-2.0) * log(u1));
    T theta = two_pi * u2;
    
    *z1 = r * cos(theta);
    *z2 = r * sin(theta);
}

// ==========================================
// 2. Global Kernels (核函数)
// ==========================================

// Uniform Kernel
template <typename T>
__global__ void RandomUniformKernel(int64_t n, uint2_ seed, T* output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 每个线程使用唯一的 Counter (基于 idx)
    uint4_ counter = {static_cast<uint32_t>(idx), 0, 0, 0}; 
    uint4_ rand_vals = philox4x32_round(counter, seed);
    output[idx] = Uint32ToReal<T>(rand_vals.x);
}

// Standard Normal Kernel
template <typename T>
__global__ void RandomStandardNormalKernel(int64_t n, uint2_ seed, T* output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 为了简化，每个线程生成一个 Normal 值，但 Box-Muller 消耗两个 uint32
    uint4_ counter = {static_cast<uint32_t>(idx), 0, 0, 0}; 
    uint4_ rand_vals = philox4x32_round(counter, seed);

    T z1, z2;
    BoxMuller(rand_vals.x, rand_vals.y, &z1, &z2);
    output[idx] = z1; 
}

// Truncated Normal Kernel
template <typename T>
__global__ void TruncatedNormalKernel(int64_t n, uint2_ seed, T* output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint4_ counter = {static_cast<uint32_t>(idx), 0, 0, 0};
    
    T val = static_cast<T>(0.0);
    bool found = false;
    
    // 拒绝采样循环 (最多尝试 100 次以防死循环)
    for(int attempt = 0; attempt < 100; ++attempt) {
        uint4_ rand_vals = philox4x32_round(counter, seed);
        
        T z1, z2;
        BoxMuller(rand_vals.x, rand_vals.y, &z1, &z2);
        
        if (abs(z1) <= static_cast<T>(2.0)) {
            val = z1; found = true; break;
        }
        if (abs(z2) <= static_cast<T>(2.0)) {
            val = z2; found = true; break;
        }
        
        // 推进计数器以获得新的随机数
        philox4x32_bump(&counter);
    }
    output[idx] = val;
}

// Uniform Int Kernel
template <typename T>
__global__ void RandomUniformIntKernel(int64_t n, uint2_ seed, T minval, T maxval, T* output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint4_ counter = {static_cast<uint32_t>(idx), 0, 0, 0};
    uint4_ rand_vals = philox4x32_round(counter, seed);

    // 处理 range
    T range = maxval - minval;
    
    // 简单的取模实现 (注意：严格来说有模偏差，但作为基础实现通用)
    if (sizeof(T) == 8) {
        uint64_t r64 = (static_cast<uint64_t>(rand_vals.x) << 32) | rand_vals.y;
        output[idx] = minval + static_cast<T>(r64 % static_cast<uint64_t>(range));
    } else {
        output[idx] = minval + static_cast<T>(rand_vals.x % static_cast<uint32_t>(range));
    }
}

// ==========================================
// 3. Host Launchers (将在 .cc 中前向声明)
// ==========================================

// 辅助函数：计算 Block 数量
inline int GetBlocks(int64_t n, int threads) {
    return (n + threads - 1) / threads;
}

// 宏定义以实例化不同类型的 Launch 函数
#define DEFINE_LAUNCH_UNIFORM(T) \
void LaunchRandomUniform_##T(void* stream, int64_t n, uint64_t seed_raw, T* output) { \
    uint2_ seed = {static_cast<uint32_t>(seed_raw), static_cast<uint32_t>(seed_raw >> 32)}; \
    int threads = 256; \
    int blocks = GetBlocks(n, threads); \
    RandomUniformKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(n, seed, output); \
}

#define DEFINE_LAUNCH_NORMAL(T) \
void LaunchRandomStandardNormal_##T(void* stream, int64_t n, uint64_t seed_raw, T* output) { \
    uint2_ seed = {static_cast<uint32_t>(seed_raw), static_cast<uint32_t>(seed_raw >> 32)}; \
    int threads = 256; \
    int blocks = GetBlocks(n, threads); \
    RandomStandardNormalKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(n, seed, output); \
}

#define DEFINE_LAUNCH_TRUNCATED(T) \
void LaunchTruncatedNormal_##T(void* stream, int64_t n, uint64_t seed_raw, T* output) { \
    uint2_ seed = {static_cast<uint32_t>(seed_raw), static_cast<uint32_t>(seed_raw >> 32)}; \
    int threads = 256; \
    int blocks = GetBlocks(n, threads); \
    TruncatedNormalKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(n, seed, output); \
}

#define DEFINE_LAUNCH_INT(T) \
void LaunchRandomUniformInt_##T(void* stream, int64_t n, uint64_t seed_raw, T minval, T maxval, T* output) { \
    uint2_ seed = {static_cast<uint32_t>(seed_raw), static_cast<uint32_t>(seed_raw >> 32)}; \
    int threads = 256; \
    int blocks = GetBlocks(n, threads); \
    RandomUniformIntKernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(n, seed, minval, maxval, output); \
}

// 显式实例化 (TensorFlow 常用的类型)
// float = float, double = double, int32 = int, int64 = long long

DEFINE_LAUNCH_UNIFORM(float)
DEFINE_LAUNCH_UNIFORM(double)

DEFINE_LAUNCH_NORMAL(float)
DEFINE_LAUNCH_NORMAL(double)

DEFINE_LAUNCH_TRUNCATED(float)
DEFINE_LAUNCH_TRUNCATED(double)

DEFINE_LAUNCH_INT(int)
DEFINE_LAUNCH_INT(int64_t)
