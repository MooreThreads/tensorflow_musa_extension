#include <musa_runtime.h>
#include <cstdint>
#include <cmath>

struct MusaPhiloxState {
    uint32_t counter[4];
    uint32_t key[2];
};

struct uint4_ { uint32_t x, y, z, w; };
struct uint2_ { uint32_t x, y; };

template <typename T>
__device__ __forceinline__ T Uint32ToReal(uint32_t x);

template <>
__device__ __forceinline__ float Uint32ToReal<float>(uint32_t x) {
    const uint32_t val = (127u << 23) | (x & 0x7fffffu);
    union { uint32_t u; float f; } p;
    p.u = val;
    return p.f - 1.0f;
}

template <>
__device__ __forceinline__ double Uint32ToReal<double>(uint32_t x) {
    return x * 2.3283064365386963e-10; 
}

__device__ __forceinline__ uint4_ skip_philox(uint4_ ctr, uint64_t skip) {
    uint32_t lo = (uint32_t)skip;
    uint32_t hi = (uint32_t)(skip >> 32);
    ctr.x += lo;
    if (ctr.x < lo) { ++hi; }
    uint32_t old_y = ctr.y;
    ctr.y += hi;
    if (ctr.y < old_y) {
        if (++ctr.z == 0) ++ctr.w;
    }
    return ctr;
}

__device__ __forceinline__ uint4_ ComputePhilox10(uint4_ ctr, uint2_ key) {
    const uint32_t M0 = 0xD2511F53;
    const uint32_t M1 = 0xCD9E8D57;
    uint2_ k = key;
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        uint64_t p0 = (uint64_t)ctr.x * M0;
        uint64_t p1 = (uint64_t)ctr.z * M1;
        uint4_ res;
        res.x = (uint32_t)(p1 >> 32) ^ ctr.y ^ k.x; 
        res.y = (uint32_t)p1;
        res.z = (uint32_t)(p0 >> 32) ^ ctr.w ^ k.y; 
        res.w = (uint32_t)p0;
        ctr = res;
        k.x += 0x9E3779B9;
        k.y += 0xBB67AE85;
    }
    return ctr;
}


template <typename T>
__device__ __forceinline__ void BoxMuller(uint32_t x, uint32_t y, T* out1, T* out2) {
    T u1 = fmaxf(Uint32ToReal<T>(x), static_cast<T>(1.0e-7));
    T r = sqrt(static_cast<T>(-2.0) * log(u1));
    T theta = static_cast<T>(6.283185307179586) * Uint32ToReal<T>(y);
    *out1 = r * sin(theta);
    *out2 = r * cos(theta);
}

// ==========================================
// Kernels
// ==========================================

template <typename T>
__global__ void RandomUniformKernel(int64_t n, MusaPhiloxState state, T* output) {
    const int kGroupSize = 4;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_thread_count = gridDim.x * blockDim.x;
    
    uint4_ ctr = {state.counter[0], state.counter[1], state.counter[2], state.counter[3]};
    uint2_ key = {state.key[0], state.key[1]};
    
    ctr = skip_philox(ctr, (uint64_t)thread_id);
    int64_t offset = (int64_t)thread_id * kGroupSize;

    while (offset < n) {
        uint4_ res = ComputePhilox10(ctr, key);
        if (offset < n) output[offset] = Uint32ToReal<T>(res.x);
        if (offset + 1 < n) output[offset + 1] = Uint32ToReal<T>(res.y);
        if (offset + 2 < n) output[offset + 2] = Uint32ToReal<T>(res.z);
        if (offset + 3 < n) output[offset + 3] = Uint32ToReal<T>(res.w);
        offset += (int64_t)total_thread_count * kGroupSize;
        ctr = skip_philox(ctr, (uint64_t)total_thread_count);
    }
}

template <typename T>
__global__ void RandomUniformIntKernel(int64_t n, MusaPhiloxState state, T minval, T maxval, T* output) {
    const int kGroupSize = 4;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_thread_count = gridDim.x * blockDim.x;
    
    uint4_ ctr = {state.counter[0], state.counter[1], state.counter[2], state.counter[3]};
    uint2_ key = {state.key[0], state.key[1]};
    
    ctr = skip_philox(ctr, (uint64_t)thread_id);
    int64_t offset = (int64_t)thread_id * kGroupSize;
    T range = maxval - minval;

    while (offset < n) {
        uint4_ res = ComputePhilox10(ctr, key);
        for (int i = 0; i < 4; ++i) {
            if (offset + i < n) {
                uint32_t val = (i == 0) ? res.x : (i == 1) ? res.y : (i == 2) ? res.z : res.w;
                output[offset + i] = minval + static_cast<T>(val % static_cast<uint32_t>(range));
            }
        }
        offset += (int64_t)total_thread_count * kGroupSize;
        ctr = skip_philox(ctr, (uint64_t)total_thread_count);
    }
}

template <typename T>
__global__ void RandomStandardNormalKernel(int64_t n, MusaPhiloxState state, T* output) {
    const int kGroupSize = 4;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_thread_count = gridDim.x * blockDim.x;
    
    uint4_ ctr = {state.counter[0], state.counter[1], state.counter[2], state.counter[3]};
    uint2_ key = {state.key[0], state.key[1]};
    
    ctr = skip_philox(ctr, (uint64_t)thread_id);
    int64_t offset = (int64_t)thread_id * kGroupSize;

    while (offset < n) {
        uint4_ res = ComputePhilox10(ctr, key);
        if (offset < n) {
            T z0, z1, z2, z3;
            BoxMuller(res.x, res.y, &z0, &z1);
            BoxMuller(res.z, res.w, &z2, &z3); 

            output[offset + 0] = z0;
            if (offset + 1 < n) output[offset + 1] = z1;
            if (offset + 2 < n) output[offset + 2] = z2;
            if (offset + 3 < n) output[offset + 3] = z3;
        }
        offset += (int64_t)total_thread_count * kGroupSize;
        ctr = skip_philox(ctr, (uint64_t)total_thread_count);
    }
}

extern "C" {
void LaunchRandomUniform_float(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, float* output) {
    RandomUniformKernel<float><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, output);
}
void LaunchRandomUniform_double(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, double* output) {
    RandomUniformKernel<double><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, output);
}
void LaunchRandomUniformInt_int(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, int minval, int maxval, int* output) {
    RandomUniformIntKernel<int><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, minval, maxval, output);
}
void LaunchRandomUniformInt_int64_t(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, int64_t minval, int64_t maxval, int64_t* output) {
    RandomUniformIntKernel<int64_t><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, minval, maxval, output);
}

void LaunchRandomStandardNormal_float(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, float* output) {
    RandomStandardNormalKernel<float><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, output);
}
void LaunchRandomStandardNormal_double(void* stream, int64_t n, int num_blocks, int block_size, MusaPhiloxState state, double* output) {
    RandomStandardNormalKernel<double><<<num_blocks, block_size, 0, (musaStream_t)stream>>>(n, state, output);
}
}