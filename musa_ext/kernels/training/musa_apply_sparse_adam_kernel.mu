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
__global__ void ResourceSparseApplyAdamKernel(
    T* __restrict__ var, T* __restrict__ m, T* __restrict__ v,
    const T* __restrict__ grad, const IndexT* __restrict__ indices,
    const T* __restrict__ lr_ptr, const T* __restrict__ beta1_ptr,
    const T* __restrict__ beta2_ptr, const T* __restrict__ epsilon_ptr,
    const T* __restrict__ beta1_power_ptr, const T* __restrict__ beta2_power_ptr,
    int64_t inner_size, int64_t indices_size) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = indices_size * inner_size;
  if (tid >= total_elements) return;

  const int64_t inner_idx = tid % inner_size;
  const int64_t indices_idx = tid / inner_size;
  const IndexT idx = indices[indices_idx];
  if (idx < 0) return;

  const int64_t var_offset = (int64_t)idx * inner_size + inner_idx;
  const int64_t grad_offset = tid;

  // Load gradient and current state values
  float g = LoadFloat(&grad[grad_offset]);
  float m_val = LoadFloat(&m[var_offset]);
  float v_val = LoadFloat(&v[var_offset]);
  float var_val = LoadFloat(&var[var_offset]);

  // Load hyperparameters
  float lr = LoadFloat(lr_ptr);
  float beta1 = LoadFloat(beta1_ptr);
  float beta2 = LoadFloat(beta2_ptr);
  float epsilon = LoadFloat(epsilon_ptr);
  float beta1_power = LoadFloat(beta1_power_ptr);
  float beta2_power = LoadFloat(beta2_power_ptr);

  // Compute bias-corrected learning rate
  // lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
  // Handle edge case when beta1_power ≈ 1.0 (initial iteration)
  float one_minus_beta1_power = 1.0f - beta1_power;
  float one_minus_beta2_power = 1.0f - beta2_power;
  float lr_t;
  if (fabsf(one_minus_beta1_power) < 1e-10f) {
    lr_t = lr;  // Initial iteration fallback
  } else {
    lr_t = lr * sqrtf(one_minus_beta2_power) / one_minus_beta1_power;
  }

  // Update m: m_t = beta1 * m + (1 - beta1) * g
  float one_minus_beta1 = 1.0f - beta1;
  float m_new = beta1 * m_val + one_minus_beta1 * g;

  // Update v: v_t = beta2 * v + (1 - beta2) * g^2
  float one_minus_beta2 = 1.0f - beta2;
  float v_new = beta2 * v_val + one_minus_beta2 * g * g;

  // Update var: var = var - lr_t * m_new / (sqrt(v_new) + epsilon)
  float v_sqrt = sqrtf(v_new);
  float var_new = var_val - lr_t * m_new / (v_sqrt + epsilon);

  // Store results
  StoreFloat(&m[var_offset], m_new);
  StoreFloat(&v[var_offset], v_new);
  StoreFloat(&var[var_offset], var_new);
}

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(n) (((n) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

template <typename T, typename IndexT>
void LaunchResourceSparseApplyAdamImpl(
    T* var, T* m, T* v, const T* grad, const IndexT* indices,
    const T* lr, const T* beta1, const T* beta2, const T* epsilon,
    const T* beta1_power, const T* beta2_power,
    int64_t inner_size, int64_t indices_size, musaStream_t stream) {
  int64_t total = inner_size * indices_size;
  if (total == 0) return;
  ResourceSparseApplyAdamKernel<T, IndexT>
      <<<OPTIMAL_BLOCKS(total), OPTIMAL_THREADS, 0, stream>>>(
          var, m, v, grad, indices, lr, beta1, beta2, epsilon,
          beta1_power, beta2_power, inner_size, indices_size);
}

#define REGISTER_SPARSE_ADAM_LAUNCHER(T, IndexT)                        \
  template void LaunchResourceSparseApplyAdamImpl<T, IndexT>(            \
      T* var, T* m, T* v, const T* grad, const IndexT* indices,          \
      const T* lr, const T* beta1, const T* beta2, const T* epsilon,     \
      const T* beta1_power, const T* beta2_power,                       \
      int64_t inner_size, int64_t indices_size, musaStream_t stream);

REGISTER_SPARSE_ADAM_LAUNCHER(float, int32);
REGISTER_SPARSE_ADAM_LAUNCHER(float, int64);
REGISTER_SPARSE_ADAM_LAUNCHER(Eigen::half, int32);
REGISTER_SPARSE_ADAM_LAUNCHER(Eigen::half, int64);
REGISTER_SPARSE_ADAM_LAUNCHER(bfloat16, int32);
REGISTER_SPARSE_ADAM_LAUNCHER(bfloat16, int64);

#undef REGISTER_SPARSE_ADAM_LAUNCHER

}  // namespace musa
}  // namespace tensorflow
