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

template <typename T>
__global__ void ApplyFtrlKernel(
    T* __restrict__ var, T* __restrict__ accum, T* __restrict__ linear,
    const T* __restrict__ grad, const T* __restrict__ lr_ptr,
    const T* __restrict__ l1_ptr, const T* __restrict__ l2_ptr,
    const T* __restrict__ l2_shrinkage_ptr,
    const T* __restrict__ lr_power_ptr, int64_t total_elements) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;

  float g = LoadFloat(&grad[tid]);
  float a = LoadFloat(&accum[tid]);
  float l = LoadFloat(&linear[tid]);
  float v = LoadFloat(&var[tid]);

  float lr = LoadFloat(lr_ptr);
  float l1 = LoadFloat(l1_ptr);
  float l2 = LoadFloat(l2_ptr);
  float l2_shrinkage = l2_shrinkage_ptr ? LoadFloat(l2_shrinkage_ptr) : 0.0f;
  float lr_power = LoadFloat(lr_power_ptr);

  float g_shrink = g + 2.0f * l2_shrinkage * v;
  float a_new = a + g * g;
  float pow_a_new = powf(a_new, -lr_power);
  float pow_a = powf(a, -lr_power);

  float sigma = (pow_a_new - pow_a) / lr;
  float l_new = l + g_shrink - sigma * v;

  float quad = pow_a_new / lr + 2.0f * l2;

  float v_new;
  if (fabsf(l_new) > l1) {
    float sign = (l_new > 0.0f) ? 1.0f : ((l_new < 0.0f) ? -1.0f : 0.0f);
    v_new = (sign * l1 - l_new) / quad;
  } else {
    v_new = 0.0f;
  }

  StoreFloat(&accum[tid], a_new);
  StoreFloat(&linear[tid], l_new);
  StoreFloat(&var[tid], v_new);
}

template <typename T, typename IndexT>
__global__ void ResourceSparseApplyFtrlKernel(
    T* __restrict__ var, T* __restrict__ accum, T* __restrict__ linear,
    const T* __restrict__ grad, const IndexT* __restrict__ indices,
    const T* __restrict__ lr_ptr, const T* __restrict__ l1_ptr,
    const T* __restrict__ l2_ptr, const T* __restrict__ l2_shrinkage_ptr,
    const T* __restrict__ lr_power_ptr, int64_t inner_size,
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
  float a = LoadFloat(&accum[var_offset]);
  float l = LoadFloat(&linear[var_offset]);
  float v = LoadFloat(&var[var_offset]);

  float lr = LoadFloat(lr_ptr);
  float l1 = LoadFloat(l1_ptr);
  float l2 = LoadFloat(l2_ptr);
  float l2_shrinkage = l2_shrinkage_ptr ? LoadFloat(l2_shrinkage_ptr) : 0.0f;
  float lr_power = LoadFloat(lr_power_ptr);

  float g_shrink = g + 2.0f * l2_shrinkage * v;
  float a_new = a + g * g;
  float pow_a_new = powf(a_new, -lr_power);
  float pow_a = powf(a, -lr_power);

  float sigma = (pow_a_new - pow_a) / lr;
  float l_new = l + g_shrink - sigma * v;

  float quad = pow_a_new / lr + 2.0f * l2;

  float v_new;
  if (fabsf(l_new) > l1) {
    float sign = (l_new > 0.0f) ? 1.0f : ((l_new < 0.0f) ? -1.0f : 0.0f);
    v_new = (sign * l1 - l_new) / quad;
  } else {
    v_new = 0.0f;
  }

  StoreFloat(&accum[var_offset], a_new);
  StoreFloat(&linear[var_offset], l_new);
  StoreFloat(&var[var_offset], v_new);
}

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(n) (((n) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

template <typename T>
void LaunchApplyFtrlImpl(T* var, T* accum, T* linear, const T* grad,
                         const T* lr, const T* l1, const T* l2,
                         const T* l2_shrinkage, const T* lr_power,
                         int64_t total_elements, musaStream_t stream) {
  if (total_elements == 0) return;
  ApplyFtrlKernel<T><<<OPTIMAL_BLOCKS(total_elements), OPTIMAL_THREADS, 0, stream>>>(
      var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power, total_elements);
}

template <typename T, typename IndexT>
void LaunchResourceSparseApplyFtrlImpl(
    T* var, T* accum, T* linear, const T* grad, const IndexT* indices,
    const T* lr, const T* l1, const T* l2, const T* l2_shrinkage,
    const T* lr_power, int64_t inner_size, int64_t indices_size,
    musaStream_t stream) {
  int64_t total = inner_size * indices_size;
  if (total == 0) return;
  ResourceSparseApplyFtrlKernel<T, IndexT><<<OPTIMAL_BLOCKS(total), OPTIMAL_THREADS, 0, stream>>>(
      var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power, inner_size, indices_size);
}

#define REGISTER_MUSA_APPLY_FTRL_LAUNCHER(T)                                 \
  template void LaunchApplyFtrlImpl<T>(                                      \
      T * var, T * accum, T * linear, const T* grad, const T* lr,            \
      const T* l1, const T* l2, const T* l2_shrinkage, const T* lr_power,    \
      int64_t total_elements, musaStream_t stream);

#define REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(T, IndexT)         \
  template void LaunchResourceSparseApplyFtrlImpl<T, IndexT>(                \
      T * var, T * accum, T * linear, const T* grad, const IndexT* indices,  \
      const T* lr, const T* l1, const T* l2, const T* l2_shrinkage,          \
      const T* lr_power, int64_t inner_size, int64_t indices_size,           \
      musaStream_t stream);

REGISTER_MUSA_APPLY_FTRL_LAUNCHER(float);
REGISTER_MUSA_APPLY_FTRL_LAUNCHER(Eigen::half);
REGISTER_MUSA_APPLY_FTRL_LAUNCHER(bfloat16);

REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(float, int32);
REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(Eigen::half, int32);
REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(bfloat16, int32);
REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(float, int64);
REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(Eigen::half, int64);
REGISTER_MUSA_RESOURCE_SPARSE_APPLY_FTRL_LAUNCHER(bfloat16, int64);

#undef REGISTER_MUSA_APPLY_FTRL_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
