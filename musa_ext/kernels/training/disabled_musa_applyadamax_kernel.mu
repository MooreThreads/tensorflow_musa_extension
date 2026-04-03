#include <stdint.h>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

template <typename T, typename AccumT>
__device__ __forceinline__ AccumT LoadValue(const T* p);

template <typename T, typename AccumT>
__device__ __forceinline__ void StoreValue(T* p, AccumT v);

template <>
__device__ __forceinline__ float LoadValue<float, float>(const float* p) {
  return *p;
}

template <>
__device__ __forceinline__ void StoreValue<float, float>(float* p, float v) {
  *p = v;
}

template <>
__device__ __forceinline__ double LoadValue<double, double>(const double* p) {
  return *p;
}

template <>
__device__ __forceinline__ void StoreValue<double, double>(double* p,
                                                           double v) {
  *p = v;
}

template <>
__device__ __forceinline__ float LoadValue<half, float>(const half* p) {
  return __half2float(*p);
}

template <>
__device__ __forceinline__ void StoreValue<half, float>(half* p, float v) {
  *p = __float2half(v);
}

template <>
__device__ __forceinline__ float LoadValue<__mt_bfloat16, float>(
    const __mt_bfloat16* p) {
  return __bfloat162float(*p);
}

template <>
__device__ __forceinline__ void StoreValue<__mt_bfloat16, float>(
    __mt_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}

template <typename T, typename AccumT>
__global__ void ApplyAdaMaxKernel(T* var, T* m, T* v, const T* grad,
                                  AccumT beta1, AccumT one_minus_beta1,
                                  AccumT beta2, AccumT epsilon, AccumT lr_t,
                                  int64_t n) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }

  const AccumT grad_value = LoadValue<T, AccumT>(grad + index);
  const AccumT m_value = LoadValue<T, AccumT>(m + index);
  const AccumT v_value = LoadValue<T, AccumT>(v + index);
  const AccumT var_value = LoadValue<T, AccumT>(var + index);

  const AccumT updated_m = beta1 * m_value + one_minus_beta1 * grad_value;
  const AccumT abs_grad =
      grad_value < static_cast<AccumT>(0) ? -grad_value : grad_value;
  const AccumT scaled_v = beta2 * v_value;
  const AccumT updated_v = scaled_v > abs_grad ? scaled_v : abs_grad;
  const AccumT updated_var =
      var_value - lr_t * updated_m / (updated_v + epsilon);

  StoreValue<T, AccumT>(m + index, updated_m);
  StoreValue<T, AccumT>(v + index, updated_v);
  StoreValue<T, AccumT>(var + index, updated_var);
}

constexpr int kThreadsPerBlock = 256;

inline int NumBlocks(int64_t n) {
  return static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

}  // namespace

extern "C" {

void LaunchApplyAdaMaxFloat(float* var, float* m, float* v, const float* grad,
                            float beta1, float one_minus_beta1, float beta2,
                            float epsilon, float lr_t, int64_t n,
                            musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  ApplyAdaMaxKernel<float, float>
      <<<NumBlocks(n), kThreadsPerBlock, 0, stream>>>(
          var, m, v, grad, beta1, one_minus_beta1, beta2, epsilon, lr_t, n);
}

void LaunchApplyAdaMaxDouble(double* var, double* m, double* v,
                             const double* grad, double beta1,
                             double one_minus_beta1, double beta2,
                             double epsilon, double lr_t, int64_t n,
                             musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  ApplyAdaMaxKernel<double, double>
      <<<NumBlocks(n), kThreadsPerBlock, 0, stream>>>(
          var, m, v, grad, beta1, one_minus_beta1, beta2, epsilon, lr_t, n);
}

void LaunchApplyAdaMaxHalf(void* var, void* m, void* v, const void* grad,
                           float beta1, float one_minus_beta1, float beta2,
                           float epsilon, float lr_t, int64_t n,
                           musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  ApplyAdaMaxKernel<half, float>
      <<<NumBlocks(n), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<half*>(var), reinterpret_cast<half*>(m),
          reinterpret_cast<half*>(v), reinterpret_cast<const half*>(grad),
          beta1, one_minus_beta1, beta2, epsilon, lr_t, n);
}

void LaunchApplyAdaMaxBFloat16(void* var, void* m, void* v, const void* grad,
                               float beta1, float one_minus_beta1,
                               float beta2, float epsilon, float lr_t,
                               int64_t n, musaStream_t stream) {
  if (n <= 0) {
    return;
  }
  ApplyAdaMaxKernel<__mt_bfloat16, float>
      <<<NumBlocks(n), kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<__mt_bfloat16*>(var),
          reinterpret_cast<__mt_bfloat16*>(m),
          reinterpret_cast<__mt_bfloat16*>(v),
          reinterpret_cast<const __mt_bfloat16*>(grad), beta1,
          one_minus_beta1, beta2, epsilon, lr_t, n);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
