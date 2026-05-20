#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {
namespace {

using bfloat16 = tensorflow::bfloat16;

constexpr int kMaxDim = 64;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr float kNegInf = -3.4028234663852886e38F;

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ float LoadValue(const float* p) { return *p; }

__device__ __forceinline__ float LoadValue(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreValue(float* p, float value) {
  *p = value;
}

__device__ __forceinline__ void StoreValue(bfloat16* p, float value) {
  const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&value);
  uint16_t* b_ptr = reinterpret_cast<uint16_t*>(p);
  *b_ptr = static_cast<uint16_t>(*f_ptr >> 16);
}

template <typename T>
__global__ void CausalAttentionGradKernel(
    const T* __restrict__ softmax, const T* __restrict__ dout,
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, const T* __restrict__ scale_ptr,
    T* __restrict__ dquery, T* __restrict__ dkey,
    T* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  __shared__ float work[kMaxDim * kMaxDim];
  __shared__ float row_dot[kMaxDim];

  for (int q = threadIdx.x; q < query_dim; q += blockDim.x) {
    row_dot[q] = 0.0f;
  }
  __syncthreads();

  const float scale = LoadValue(scale_ptr);
  const int64_t softmax_base = group * static_cast<int64_t>(query_dim) * key_dim;
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * value_dim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * value_dim;
  const int64_t v_base = k_base;

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float ds = 0.0f;
    if (k <= last_unmasked_key) {
      for (int d = 0; d < value_dim; ++d) {
        ds += LoadValue(dout + q_base + static_cast<int64_t>(q) * value_dim + d) *
              LoadValue(value + v_base + static_cast<int64_t>(k) * value_dim + d);
      }
      const float s = LoadValue(softmax + softmax_base + idx);
      atomicAdd(&row_dot[q], ds * s);
    }
    work[idx] = ds;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float dlogits = 0.0f;
    if (k <= last_unmasked_key) {
      const float s = LoadValue(softmax + softmax_base + idx);
      dlogits = (work[idx] - row_dot[q]) * s * scale;
    }
    work[idx] = dlogits;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * value_dim; idx += blockDim.x) {
    const int q = idx / value_dim;
    const int d = idx - q * value_dim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += work[q * key_dim + k] *
             LoadValue(key + k_base + static_cast<int64_t>(k) * value_dim + d);
    }
    StoreValue(dquery + q_base + static_cast<int64_t>(q) * value_dim + d, sum);
  }

  for (int idx = threadIdx.x; idx < key_dim * value_dim; idx += blockDim.x) {
    const int k = idx / value_dim;
    const int d = idx - k * value_dim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += work[q * key_dim + k] *
                  LoadValue(query + q_base + static_cast<int64_t>(q) * value_dim + d);
        dv_sum += LoadValue(softmax + softmax_base + static_cast<int64_t>(q) * key_dim + k) *
                  LoadValue(dout + q_base + static_cast<int64_t>(q) * value_dim + d);
      }
    }
    StoreValue(dkey + k_base + static_cast<int64_t>(k) * value_dim + d, dk_sum);
    StoreValue(dvalue + v_base + static_cast<int64_t>(k) * value_dim + d, dv_sum);
  }
}


template <typename T>
__global__ void CausalAttentionGradDh8RecomputeKernel(
    const T* __restrict__ softmax, const T* __restrict__ dout,
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, const T* __restrict__ scale_ptr,
    T* __restrict__ dquery, T* __restrict__ dkey,
    T* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim) {
  (void)softmax;
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  constexpr int kValueDim = 8;
  constexpr int kMaxRecomputeDim = 40;
  constexpr int kMaxRecomputeElems = kMaxRecomputeDim * kMaxRecomputeDim;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  __shared__ float dlogits[kMaxRecomputeElems];
  __shared__ float probs[kMaxRecomputeElems];

  const float scale = LoadValue(scale_ptr);
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * kValueDim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * kValueDim;
  const int64_t v_base = k_base;

  for (int q = warp_id; q < query_dim; q += kWarpsPerBlock) {
    const int last_unmasked_key = key_dim - query_dim + q;
    const int key_idx0 = lane;
    const int key_idx1 = lane + kWarpSize;
    const int64_t q_off = q_base + static_cast<int64_t>(q) * kValueDim;

    const float q0 = LoadValue(query + q_off + 0);
    const float q1 = LoadValue(query + q_off + 1);
    const float q2 = LoadValue(query + q_off + 2);
    const float q3 = LoadValue(query + q_off + 3);
    const float q4 = LoadValue(query + q_off + 4);
    const float q5 = LoadValue(query + q_off + 5);
    const float q6 = LoadValue(query + q_off + 6);
    const float q7 = LoadValue(query + q_off + 7);

    auto dot_key = [&](int key_idx) {
      const T* key_ptr =
          key + k_base + static_cast<int64_t>(key_idx) * kValueDim;
      return (q0 * LoadValue(key_ptr + 0) + q1 * LoadValue(key_ptr + 1) +
              q2 * LoadValue(key_ptr + 2) + q3 * LoadValue(key_ptr + 3) +
              q4 * LoadValue(key_ptr + 4) + q5 * LoadValue(key_ptr + 5) +
              q6 * LoadValue(key_ptr + 6) + q7 * LoadValue(key_ptr + 7)) *
             scale;
    };

    const bool valid0 = key_idx0 < key_dim && key_idx0 <= last_unmasked_key;
    const bool valid1 = key_idx1 < key_dim && key_idx1 <= last_unmasked_key;
    const float x0 = valid0 ? dot_key(key_idx0) : kNegInf;
    const float x1 = valid1 ? dot_key(key_idx1) : kNegInf;
    const float row_max = WarpReduceMax(fmaxf(x0, x1));
    const float e0 = valid0 ? __expf(x0 - row_max) : 0.0f;
    const float e1 = valid1 ? __expf(x1 - row_max) : 0.0f;
    const float row_sum = WarpReduceSum(e0 + e1);
    const float inv_sum = 1.0f / row_sum;
    const float p0 = e0 * inv_sum;
    const float p1 = e1 * inv_sum;

    const T* dout_ptr = dout + q_off;
    auto dot_value = [&](int key_idx) {
      const T* value_ptr =
          value + v_base + static_cast<int64_t>(key_idx) * kValueDim;
      return LoadValue(dout_ptr + 0) * LoadValue(value_ptr + 0) +
             LoadValue(dout_ptr + 1) * LoadValue(value_ptr + 1) +
             LoadValue(dout_ptr + 2) * LoadValue(value_ptr + 2) +
             LoadValue(dout_ptr + 3) * LoadValue(value_ptr + 3) +
             LoadValue(dout_ptr + 4) * LoadValue(value_ptr + 4) +
             LoadValue(dout_ptr + 5) * LoadValue(value_ptr + 5) +
             LoadValue(dout_ptr + 6) * LoadValue(value_ptr + 6) +
             LoadValue(dout_ptr + 7) * LoadValue(value_ptr + 7);
    };

    const float ds0 = valid0 ? dot_value(key_idx0) : 0.0f;
    const float ds1 = valid1 ? dot_value(key_idx1) : 0.0f;
    const float row_dot = WarpReduceSum(ds0 * p0 + ds1 * p1);
    const int row_base = q * key_dim;
    if (key_idx0 < key_dim) {
      probs[row_base + key_idx0] = p0;
      dlogits[row_base + key_idx0] =
          valid0 ? (ds0 - row_dot) * p0 * scale : 0.0f;
    }
    if (key_idx1 < key_dim) {
      probs[row_base + key_idx1] = p1;
      dlogits[row_base + key_idx1] =
          valid1 ? (ds1 - row_dot) * p1 * scale : 0.0f;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * kValueDim; idx += blockDim.x) {
    const int q = idx / kValueDim;
    const int d = idx - q * kValueDim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += dlogits[q * key_dim + k] *
             LoadValue(key + k_base + static_cast<int64_t>(k) * kValueDim + d);
    }
    StoreValue(dquery + q_base + static_cast<int64_t>(q) * kValueDim + d, sum);
  }

  for (int idx = threadIdx.x; idx < key_dim * kValueDim; idx += blockDim.x) {
    const int k = idx / kValueDim;
    const int d = idx - k * kValueDim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += dlogits[q * key_dim + k] *
                  LoadValue(query + q_base + static_cast<int64_t>(q) * kValueDim + d);
        dv_sum += probs[q * key_dim + k] *
                  LoadValue(dout + q_base + static_cast<int64_t>(q) * kValueDim + d);
      }
    }
    StoreValue(dkey + k_base + static_cast<int64_t>(k) * kValueDim + d, dk_sum);
    StoreValue(dvalue + v_base + static_cast<int64_t>(k) * kValueDim + d, dv_sum);
  }
}

template <typename T, int kValueDim>
__global__ void CausalAttentionGradFixedValueDimKernel(
    const T* __restrict__ softmax, const T* __restrict__ dout,
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, const T* __restrict__ scale_ptr,
    T* __restrict__ dquery, T* __restrict__ dkey,
    T* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  __shared__ float work[kMaxDim * kMaxDim];
  __shared__ float row_dot[kMaxDim];

  for (int q = threadIdx.x; q < query_dim; q += blockDim.x) {
    row_dot[q] = 0.0f;
  }
  __syncthreads();

  const float scale = LoadValue(scale_ptr);
  const int64_t softmax_base = group * static_cast<int64_t>(query_dim) * key_dim;
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * kValueDim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * kValueDim;
  const int64_t v_base = k_base;

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float ds = 0.0f;
    if (k <= last_unmasked_key) {
#pragma unroll
      for (int d = 0; d < kValueDim; ++d) {
        ds += LoadValue(dout + q_base + static_cast<int64_t>(q) * kValueDim + d) *
              LoadValue(value + v_base + static_cast<int64_t>(k) * kValueDim + d);
      }
      const float s = LoadValue(softmax + softmax_base + idx);
      atomicAdd(&row_dot[q], ds * s);
    }
    work[idx] = ds;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float dlogits = 0.0f;
    if (k <= last_unmasked_key) {
      const float s = LoadValue(softmax + softmax_base + idx);
      dlogits = (work[idx] - row_dot[q]) * s * scale;
    }
    work[idx] = dlogits;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * kValueDim; idx += blockDim.x) {
    const int q = idx / kValueDim;
    const int d = idx - q * kValueDim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += work[q * key_dim + k] *
             LoadValue(key + k_base + static_cast<int64_t>(k) * kValueDim + d);
    }
    StoreValue(dquery + q_base + static_cast<int64_t>(q) * kValueDim + d, sum);
  }

  for (int idx = threadIdx.x; idx < key_dim * kValueDim; idx += blockDim.x) {
    const int k = idx / kValueDim;
    const int d = idx - k * kValueDim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += work[q * key_dim + k] *
                  LoadValue(query + q_base + static_cast<int64_t>(q) * kValueDim + d);
        dv_sum += LoadValue(softmax + softmax_base + static_cast<int64_t>(q) * key_dim + k) *
                  LoadValue(dout + q_base + static_cast<int64_t>(q) * kValueDim + d);
      }
    }
    StoreValue(dkey + k_base + static_cast<int64_t>(k) * kValueDim + d, dk_sum);
    StoreValue(dvalue + v_base + static_cast<int64_t>(k) * kValueDim + d, dv_sum);
  }
}

}  // namespace

extern "C" void LaunchMusaCausalAttentionGradFloat(
    const float* softmax, const float* dout, const float* query,
    const float* key, const float* value, const float* scale, float* dquery,
    float* dkey, float* dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim, musaStream_t stream) {
  if (groups <= 0 || query_dim <= 0 || key_dim <= 0 || value_dim <= 0 ||
      query_dim > key_dim || query_dim > kMaxDim || key_dim > kMaxDim ||
      value_dim > kMaxDim) {
    return;
  }
  if (value_dim == 8 && query_dim <= 40 && key_dim <= 40) {
    CausalAttentionGradDh8RecomputeKernel<float><<<groups, kThreads, 0, stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  if (value_dim == 8) {
    CausalAttentionGradFixedValueDimKernel<float, 8><<<groups, kThreads, 0, stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  CausalAttentionGradKernel<float><<<groups, kThreads, 0, stream>>>(
      softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
      query_dim, key_dim, value_dim);
}

extern "C" void LaunchMusaCausalAttentionGradBFloat16(
    const bfloat16* softmax, const bfloat16* dout, const bfloat16* query,
    const bfloat16* key, const bfloat16* value, const bfloat16* scale,
    bfloat16* dquery, bfloat16* dkey, bfloat16* dvalue, int64_t groups,
    int query_dim, int key_dim, int value_dim, musaStream_t stream) {
  if (groups <= 0 || query_dim <= 0 || key_dim <= 0 || value_dim <= 0 ||
      query_dim > key_dim || query_dim > kMaxDim || key_dim > kMaxDim ||
      value_dim > kMaxDim) {
    return;
  }
  if (value_dim == 8 && query_dim <= 40 && key_dim <= 40) {
    CausalAttentionGradDh8RecomputeKernel<bfloat16><<<groups, kThreads, 0, stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  if (value_dim == 8) {
    CausalAttentionGradFixedValueDimKernel<bfloat16, 8><<<groups, kThreads, 0,
                                                         stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  CausalAttentionGradKernel<bfloat16><<<groups, kThreads, 0, stream>>>(
      softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
      query_dim, key_dim, value_dim);
}

}  // namespace musa
}  // namespace tensorflow
