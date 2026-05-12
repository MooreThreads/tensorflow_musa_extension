#include <stdint.h>

#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;
constexpr int64_t kSmallOutputCols = 32;

inline int BlocksFor(int64_t work_items) {
  const int64_t blocks64 =
      (work_items + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(blocks64 > kMaxBlocks ? kMaxBlocks : blocks64);
}

template <typename T>
struct AccumType {
  using Type = T;
};

template <>
struct AccumType<bfloat16> {
  using Type = float;
};

template <typename T>
struct UseScatterKernel {
  static constexpr bool value = true;
};

template <>
struct UseScatterKernel<bfloat16> {
  static constexpr bool value = false;
};

template <typename T>
__device__ __forceinline__ typename AccumType<T>::Type LoadAsAccum(
    const T* value) {
  return *value;
}

template <>
__device__ __forceinline__ float LoadAsAccum<bfloat16>(
    const bfloat16* value) {
  float result = 0.0f;
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(value);
  *reinterpret_cast<uint32_t*>(&result) = static_cast<uint32_t>(bits) << 16;
  return result;
}

template <typename T>
__device__ __forceinline__ void AtomicAddValue(
    T* address, typename AccumType<T>::Type value) {
  atomicAdd(address, value);
}

template <typename T>
__device__ __forceinline__ void StoreFromAccum(
    T* address, typename AccumType<T>::Type value) {
  *address = value;
}

__device__ __forceinline__ float BFloat16BitsToFloat(uint16_t bits) {
  float result = 0.0f;
  *reinterpret_cast<uint32_t*>(&result) = static_cast<uint32_t>(bits) << 16;
  return result;
}

__device__ __forceinline__ uint16_t FloatToBFloat16Bits(float value) {
  const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
  return static_cast<uint16_t>(bits >> 16);
}

template <>
__device__ __forceinline__ void AtomicAddValue<bfloat16>(bfloat16* address,
                                                         float value) {
  uintptr_t byte_address = reinterpret_cast<uintptr_t>(address);
  unsigned int* word_address =
      reinterpret_cast<unsigned int*>(byte_address & ~uintptr_t(0x3));
  const unsigned int shift = (byte_address & 0x2) ? 16 : 0;
  const unsigned int mask = 0xffffu << shift;

  unsigned int old = *word_address;
  unsigned int assumed = 0;
  do {
    assumed = old;
    const uint16_t old_bits =
        static_cast<uint16_t>((assumed >> shift) & 0xffffu);
    const float new_value = BFloat16BitsToFloat(old_bits) + value;
    const unsigned int new_bits =
        static_cast<unsigned int>(FloatToBFloat16Bits(new_value)) << shift;
    const unsigned int desired = (assumed & ~mask) | new_bits;
    old = atomicCAS(word_address, assumed, desired);
  } while (old != assumed);
}

template <>
__device__ __forceinline__ void StoreFromAccum<bfloat16>(bfloat16* address,
                                                         float value) {
  const uint16_t bits = FloatToBFloat16Bits(value);
  *reinterpret_cast<uint16_t*>(address) = bits;
}

template <bool ADJ_A>
__device__ __forceinline__ void DecodeSparseIndex(
    const int64* indices, int64_t nz, int64_t a_rows, int64_t a_cols,
    int64_t* out_row, int64_t* contract_col, bool* valid) {
  const int64_t raw_row = indices[nz * 2];
  const int64_t raw_col = indices[nz * 2 + 1];
  *valid = raw_row >= 0 && raw_row < a_rows && raw_col >= 0 &&
           raw_col < a_cols;
  *out_row = ADJ_A ? raw_col : raw_row;
  *contract_col = ADJ_A ? raw_row : raw_col;
}

template <bool ADJ_B>
__device__ __forceinline__ int64_t DenseBOffset(int64_t contract_col,
                                                int64_t out_col,
                                                int64_t b_cols) {
  return ADJ_B ? out_col * b_cols + contract_col
               : contract_col * b_cols + out_col;
}

template <typename T, bool ADJ_A, bool ADJ_B>
__global__ void SparseDenseMatMulScalarKernel(
    const int64* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, T* __restrict__ output, int64_t nnz,
    int64_t a_rows, int64_t a_cols, int64_t b_cols) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t nz =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       nz < nnz; nz += grid_stride) {
    int64_t out_row = 0;
    int64_t contract_col = 0;
    bool valid = false;
    DecodeSparseIndex<ADJ_A>(a_indices, nz, a_rows, a_cols, &out_row,
                             &contract_col, &valid);
    if (!valid) continue;

    const auto a = LoadAsAccum<T>(&a_values[nz]);
    const auto dense =
        LoadAsAccum<T>(&b[DenseBOffset<ADJ_B>(contract_col, 0, b_cols)]);
    AtomicAddValue<T>(&output[out_row], a * dense);
  }
}

template <typename T, bool ADJ_A, bool ADJ_B>
__global__ void SparseDenseMatMulSmallNKernel(
    const int64* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, T* __restrict__ output, int64_t nnz,
    int64_t a_rows, int64_t a_cols, int64_t b_cols, int64_t out_cols) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t nz =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       nz < nnz; nz += grid_stride) {
    int64_t out_row = 0;
    int64_t contract_col = 0;
    bool valid = false;
    DecodeSparseIndex<ADJ_A>(a_indices, nz, a_rows, a_cols, &out_row,
                             &contract_col, &valid);
    if (!valid) continue;

    const auto a = LoadAsAccum<T>(&a_values[nz]);
    T* output_row = output + out_row * out_cols;

    #pragma unroll 1
    for (int64_t out_col = 0; out_col < out_cols; ++out_col) {
      const auto dense =
          LoadAsAccum<T>(&b[DenseBOffset<ADJ_B>(contract_col, out_col,
                                                b_cols)]);
      AtomicAddValue<T>(&output_row[out_col], a * dense);
    }
  }
}

template <typename T, bool ADJ_A, bool ADJ_B>
__global__ void SparseDenseMatMulElementKernel(
    const int64* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, T* __restrict__ output, int64_t total_work,
    int64_t a_rows, int64_t a_cols, int64_t b_cols, int64_t out_cols) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t tid =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total_work; tid += grid_stride) {
    const int64_t out_col = tid % out_cols;
    const int64_t nz = tid / out_cols;

    int64_t out_row = 0;
    int64_t contract_col = 0;
    bool valid = false;
    DecodeSparseIndex<ADJ_A>(a_indices, nz, a_rows, a_cols, &out_row,
                             &contract_col, &valid);
    if (!valid) continue;

    const auto a = LoadAsAccum<T>(&a_values[nz]);
    const auto dense =
        LoadAsAccum<T>(&b[DenseBOffset<ADJ_B>(contract_col, out_col, b_cols)]);
    AtomicAddValue<T>(&output[out_row * out_cols + out_col], a * dense);
  }
}

template <typename T, bool ADJ_A, bool ADJ_B>
__global__ void SparseDenseMatMulOutputScanKernel(
    const int64* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, T* __restrict__ output, int64_t nnz,
    int64_t total_output, int64_t a_rows, int64_t a_cols, int64_t b_cols,
    int64_t out_cols) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t tid =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total_output; tid += grid_stride) {
    const int64_t out_col = tid % out_cols;
    const int64_t target_row = tid / out_cols;
    typename AccumType<T>::Type sum = typename AccumType<T>::Type();

    #pragma unroll 1
    for (int64_t nz = 0; nz < nnz; ++nz) {
      int64_t out_row = 0;
      int64_t contract_col = 0;
      bool valid = false;
      DecodeSparseIndex<ADJ_A>(a_indices, nz, a_rows, a_cols, &out_row,
                               &contract_col, &valid);
      if (!valid || out_row != target_row) continue;

      const auto a = LoadAsAccum<T>(&a_values[nz]);
      const auto dense =
          LoadAsAccum<T>(&b[DenseBOffset<ADJ_B>(contract_col, out_col,
                                                b_cols)]);
      sum += a * dense;
    }
    StoreFromAccum<T>(&output[tid], sum);
  }
}

template <typename T, bool ADJ_A, bool ADJ_B>
musaError_t LaunchSparseTensorDenseMatMulImpl(
    const int64* a_indices, const T* a_values, const T* b, T* output,
    int64_t nnz, int64_t a_rows, int64_t a_cols, int64_t b_cols,
    int64_t out_rows, int64_t out_cols, musaStream_t stream) {
  if (nnz <= 0 || out_cols <= 0) return musaSuccess;

  if (!UseScatterKernel<T>::value) {
    const int64_t total_output = out_rows * out_cols;
    SparseDenseMatMulOutputScanKernel<T, ADJ_A, ADJ_B>
        <<<BlocksFor(total_output), kThreadsPerBlock, 0, stream>>>(
            a_indices, a_values, b, output, nnz, total_output, a_rows, a_cols,
            b_cols, out_cols);
    return musaGetLastError();
  }

  if (out_cols == 1) {
    SparseDenseMatMulScalarKernel<T, ADJ_A, ADJ_B>
        <<<BlocksFor(nnz), kThreadsPerBlock, 0, stream>>>(
            a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols);
    return musaGetLastError();
  }

  if (out_cols <= kSmallOutputCols) {
    SparseDenseMatMulSmallNKernel<T, ADJ_A, ADJ_B>
        <<<BlocksFor(nnz), kThreadsPerBlock, 0, stream>>>(
            a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols,
            out_cols);
    return musaGetLastError();
  }

  const int64_t total_work = nnz * out_cols;
  SparseDenseMatMulElementKernel<T, ADJ_A, ADJ_B>
      <<<BlocksFor(total_work), kThreadsPerBlock, 0, stream>>>(
          a_indices, a_values, b, output, total_work, a_rows, a_cols, b_cols,
          out_cols);
  return musaGetLastError();
}

}  // namespace

template <typename T>
musaError_t LaunchSparseTensorDenseMatMul(
    const int64* a_indices, const T* a_values, const T* b, T* output,
    int64_t nnz, int64_t a_rows, int64_t a_cols, int64_t b_rows,
    int64_t b_cols, int64_t out_rows, int64_t out_cols, bool adjoint_a,
    bool adjoint_b, musaStream_t stream) {
  (void)b_rows;

  if (adjoint_a) {
    if (adjoint_b) {
      return LaunchSparseTensorDenseMatMulImpl<T, true, true>(
          a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols,
          out_rows, out_cols, stream);
    }
    return LaunchSparseTensorDenseMatMulImpl<T, true, false>(
        a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols, out_rows,
        out_cols, stream);
  }

  if (adjoint_b) {
    return LaunchSparseTensorDenseMatMulImpl<T, false, true>(
        a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols, out_rows,
        out_cols, stream);
  }
  return LaunchSparseTensorDenseMatMulImpl<T, false, false>(
      a_indices, a_values, b, output, nnz, a_rows, a_cols, b_cols, out_rows,
      out_cols, stream);
}

template musaError_t LaunchSparseTensorDenseMatMul<float>(
    const int64*, const float*, const float*, float*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool, musaStream_t);
template musaError_t LaunchSparseTensorDenseMatMul<double>(
    const int64*, const double*, const double*, double*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool, musaStream_t);
template musaError_t LaunchSparseTensorDenseMatMul<int32>(
    const int64*, const int32*, const int32*, int32*, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool, musaStream_t);
template musaError_t LaunchSparseTensorDenseMatMul<bfloat16>(
    const int64*, const bfloat16*, const bfloat16*, bfloat16*, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool,
    musaStream_t);

}  // namespace musa
}  // namespace tensorflow
