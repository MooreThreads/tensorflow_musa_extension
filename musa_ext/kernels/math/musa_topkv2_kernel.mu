#include <float.h>
#include <math.h>
#include <stdint.h>

#include <musa_fp16.h>
#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kTopKMaxK = 1024;

// ===================== Load / Store Helpers =====================

__device__ __forceinline__ float LoadAsFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFromFloat(float v, float* p) { *p = v; }

__device__ __forceinline__ float LoadAsFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFromFloat(float v, Eigen::half* p) {
  __half hv = __float2half(v);
  *reinterpret_cast<__half*>(p) = hv;
}

__device__ __forceinline__ float LoadAsFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreFromFloat(float v, bfloat16* p) {
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&v);
  uint16_t* dst = reinterpret_cast<uint16_t*>(p);
  *dst = static_cast<uint16_t>((*src) >> 16);
}

template <typename Tidx>
__device__ __forceinline__ Tidx CastIndex(int v) {
  return static_cast<Tidx>(v);
}

template <typename T>
__device__ __forceinline__ bool IsBetterCandidate(float cand_val, int cand_idx,
                                                  float best_val,
                                                  int best_idx) {
  return (cand_val > best_val) ||
         ((cand_val == best_val) && (cand_idx < best_idx));
}

template <typename T, typename Tidx>
__global__ void TopKV2Kernel(const T* input, T* values, Tidx* indices, int rows,
                             int cols, int k, bool sorted) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  if (threadIdx.x != 0) return;

  const T* row_in = input + row * cols;
  T* row_out_val = values + row * k;
  Tidx* row_out_idx = indices + row * k;

  int selected_idx[kTopKMaxK];

  #pragma unroll 1
  for (int out_i = 0; out_i < k; ++out_i) {
    float best_val = -FLT_MAX;
    int best_idx = -1;

    for (int col = 0; col < cols; ++col) {
      bool already_selected = false;
      #pragma unroll 1
      for (int t = 0; t < out_i; ++t) {
        if (selected_idx[t] == col) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) continue;

      const float v = LoadAsFloat(&row_in[col]);
      if (best_idx < 0 || IsBetterCandidate<T>(v, col, best_val, best_idx)) {
        best_val = v;
        best_idx = col;
      }
    }

    selected_idx[out_i] = best_idx;
    StoreFromFloat(best_val, &row_out_val[out_i]);
    row_out_idx[out_i] = CastIndex<Tidx>(best_idx);
  }

  if (!sorted && k > 1) {
    // TensorFlow only guarantees sorted order when sorted=true.
    // Returning sorted output even when sorted=false is acceptable.
  }
}

}  // namespace

template <typename T, typename Tidx>
void LaunchTopKV2(const T* input, T* values, Tidx* indices, int rows, int cols,
                  int k, bool sorted, musaStream_t stream) {
  if (rows <= 0 || cols <= 0 || k <= 0) return;

  if (k > kTopKMaxK) {
    return;
  }

  const int blocks = rows;
  const int threads = 1;
  TopKV2Kernel<T, Tidx><<<blocks, threads, 0, stream>>>(input, values, indices,
                                                         rows, cols, k, sorted);

  musaError_t err = musaGetLastError();
  (void)err;
}

template void LaunchTopKV2<float, int32>(const float*, float*, int32*, int, int,
                                         int, bool, musaStream_t);
template void LaunchTopKV2<float, int64>(const float*, float*, int64*, int, int,
                                         int, bool, musaStream_t);
template void LaunchTopKV2<Eigen::half, int32>(const Eigen::half*, Eigen::half*,
                                               int32*, int, int, int, bool,
                                               musaStream_t);
template void LaunchTopKV2<Eigen::half, int64>(const Eigen::half*, Eigen::half*,
                                               int64*, int, int, int, bool,
                                               musaStream_t);
template void LaunchTopKV2<bfloat16, int32>(const bfloat16*, bfloat16*, int32*,
                                            int, int, int, bool, musaStream_t);
template void LaunchTopKV2<bfloat16, int64>(const bfloat16*, bfloat16*, int64*,
                                            int, int, int, bool, musaStream_t);

}  // namespace musa
}  // namespace tensorflow