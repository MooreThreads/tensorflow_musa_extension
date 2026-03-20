#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {
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
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  uint32_t* f_ptr = (uint32_t*)&v;
  uint16_t b_val = (*f_ptr) >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}
}  // namespace

template <typename T, typename IndexT>
__global__ void ResourceSparseApplyAdaGradV2Kernel(
    T* __restrict__ var, T* __restrict__ accum, const T* __restrict__ lr,
    const T* __restrict__ epsilon, const T* __restrict__ grad,
    const IndexT* __restrict__ indices, int64_t inner_size,
    int64_t indices_size) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = indices_size * inner_size;

  if (tid >= total_elements) return;

  const int64_t inner_idx = tid % inner_size;
  const int64_t indices_idx = tid / inner_size;
  const IndexT idx = indices[indices_idx];

  if (idx < 0) return;

  const int64_t var_offset = (int64_t)idx * inner_size + inner_idx;
  const int64_t grad_offset = tid;

  float g = LoadFloat(&grad[grad_offset]);
  float a = LoadFloat(&accum[var_offset]) + g * g;
  StoreFloat(&accum[var_offset], a);

  StoreFloat(&var[var_offset],
             LoadFloat(&var[var_offset]) -
                 LoadFloat(lr) * g / (sqrtf(a) + LoadFloat(epsilon)));
}

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(n) (((n) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

#define DEFINE_V2_LAUNCHER(T, IndexT, Name)                                    \
  void Name(T* var, T* accum, const T* lr, const T* epsilon, const T* grad,    \
            const IndexT* indices, int64_t inner_size, int64_t indices_size,   \
            musaStream_t stream) {                                             \
    int64_t total = inner_size * indices_size;                                 \
    if (total == 0) return;                                                    \
    ResourceSparseApplyAdaGradV2Kernel<T, IndexT>                              \
        <<<OPTIMAL_BLOCKS(total), OPTIMAL_THREADS, 0, stream>>>(               \
            var, accum, lr, epsilon, grad, indices, inner_size, indices_size); \
  }

extern "C" {
DEFINE_V2_LAUNCHER(float, int32_t, LaunchResourceSparseApplyAdaGradV2FloatInt32)
DEFINE_V2_LAUNCHER(float, int64_t, LaunchResourceSparseApplyAdaGradV2FloatInt64)

DEFINE_V2_LAUNCHER(Eigen::half, int32_t,
                   LaunchResourceSparseApplyAdaGradV2HalfInt32)
DEFINE_V2_LAUNCHER(Eigen::half, int64_t,
                   LaunchResourceSparseApplyAdaGradV2HalfInt64)

DEFINE_V2_LAUNCHER(bfloat16, int32_t,
                   LaunchResourceSparseApplyAdaGradV2BFloat16Int32)
DEFINE_V2_LAUNCHER(bfloat16, int64_t,
                   LaunchResourceSparseApplyAdaGradV2BFloat16Int64)
}

}  // namespace musa
}  // namespace tensorflow