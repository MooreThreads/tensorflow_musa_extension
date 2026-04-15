#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

#include "musa_fused_elementwise_kernel.h"

using bfloat16 = tensorflow::bfloat16;

namespace tensorflow {
namespace musa {

template <typename T>
struct MusaFusedElementwiseAccumType {
  using type = float;
};

template <>
struct MusaFusedElementwiseAccumType<double> {
  using type = double;
};

__device__ __forceinline__ float LoadElement(const float* p) { return *p; }
__device__ __forceinline__ double LoadElement(const double* p) { return *p; }

__device__ __forceinline__ float LoadElement(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ float LoadElement(const bfloat16* p) {
  float result = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&result);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return result;
}

__device__ __forceinline__ void StoreElement(float* p, float v) { *p = v; }
__device__ __forceinline__ void StoreElement(double* p, double v) { *p = v; }

__device__ __forceinline__ void StoreElement(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ void StoreElement(bfloat16* p, float v) {
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&v);
  uint16_t b_val = *f_ptr >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__device__ __forceinline__ const T* GetInputPointer(
    MusaFusedElementwiseInlinePointers inputs, int input_index) {
  return reinterpret_cast<const T*>(inputs.ptrs[input_index]);
}

template <typename T>
__device__ __forceinline__ typename MusaFusedElementwiseAccumType<T>::type
LoadInputValue(MusaFusedElementwiseInlinePointers inputs, int input_index,
               int offset) {
  const T* ptr = GetInputPointer<T>(inputs, input_index);
  return static_cast<typename MusaFusedElementwiseAccumType<T>::type>(
      LoadElement(ptr + offset));
}

template <typename AccT>
__device__ __forceinline__ AccT ApplyUnary(int opcode, AccT x) {
  switch (opcode) {
    case kOpcodeExp:
      return exp(x);
    case kOpcodeLog:
      return log(x);
    case kOpcodeRsqrt:
      return AccT(1) / sqrt(x);
    case kOpcodeRelu:
      return x > AccT(0) ? x : AccT(0);
    case kOpcodeTanh:
      return tanh(x);
    case kOpcodeSigmoid:
      return AccT(1) / (AccT(1) + exp(-x));
    case kOpcodeNeg:
      return -x;
    default:
      return x;
  }
}

template <typename AccT>
__device__ __forceinline__ AccT ApplyBinary(int opcode, AccT lhs, AccT rhs) {
  switch (opcode) {
    case kOpcodeAdd:
      return lhs + rhs;
    case kOpcodeSub:
      return lhs - rhs;
    case kOpcodeMul:
      return lhs * rhs;
    case kOpcodeRealDiv:
      return lhs / rhs;
    case kOpcodeMaximum:
      return lhs > rhs ? lhs : rhs;
    case kOpcodeMinimum:
      return lhs < rhs ? lhs : rhs;
    default:
      return lhs;
  }
}

template <typename T>
__global__ void MusaFusedElementwiseKernel(
    MusaFusedElementwiseInlinePointers inputs, T* output,
    MusaFusedElementwiseConfig config, int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  int coords[kMusaFusedElementwiseMaxDims] = {0};
  int remaining = idx;
  for (int dim = config.rank - 1; dim >= 0; --dim) {
    const int size = config.dims[dim];
    coords[dim] = remaining % size;
    remaining /= size;
  }

  int input_offsets[kMusaFusedElementwiseMaxInputs] = {0};
  for (int input_idx = 0; input_idx < config.num_inputs; ++input_idx) {
    int offset = 0;
    for (int dim = 0; dim < config.rank; ++dim) {
      offset += coords[dim] * config.input_strides[input_idx][dim];
    }
    input_offsets[input_idx] = offset;
  }

  using AccT = typename MusaFusedElementwiseAccumType<T>::type;
  AccT prev = AccT(0);

  for (int step = 0; step < config.num_steps; ++step) {
    const int arg0_kind = config.step_arg_kind[step][0];
    const int arg0_input = config.step_arg_input[step][0];
    const AccT arg0 =
        arg0_kind == kOperandPrev
            ? prev
            : LoadInputValue<T>(inputs, arg0_input, input_offsets[arg0_input]);

    if (config.step_arity[step] == 1) {
      prev = ApplyUnary(config.step_opcode[step], arg0);
      continue;
    }

    const int arg1_kind = config.step_arg_kind[step][1];
    const int arg1_input = config.step_arg_input[step][1];
    const AccT arg1 =
        arg1_kind == kOperandPrev
            ? prev
            : LoadInputValue<T>(inputs, arg1_input, input_offsets[arg1_input]);
    prev = ApplyBinary(config.step_opcode[step], arg0, arg1);
  }

  StoreElement(output + idx, prev);
}

template <typename T>
void LaunchMusaFusedElementwiseKernel(
    MusaFusedElementwiseInlinePointers inputs, T* output,
    const MusaFusedElementwiseConfig& config, int total_elements,
    musaStream_t stream) {
  if (total_elements <= 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (total_elements + block_size - 1) / block_size;
  MusaFusedElementwiseKernel<T><<<grid_size, block_size, 0, stream>>>(
      inputs, output, config, total_elements);
}

template void LaunchMusaFusedElementwiseKernel<float>(
    MusaFusedElementwiseInlinePointers, float*,
    const MusaFusedElementwiseConfig&, int, musaStream_t);
template void LaunchMusaFusedElementwiseKernel<double>(
    MusaFusedElementwiseInlinePointers, double*,
    const MusaFusedElementwiseConfig&, int, musaStream_t);
template void LaunchMusaFusedElementwiseKernel<Eigen::half>(
    MusaFusedElementwiseInlinePointers, Eigen::half*,
    const MusaFusedElementwiseConfig&, int, musaStream_t);
template void LaunchMusaFusedElementwiseKernel<bfloat16>(
    MusaFusedElementwiseInlinePointers, bfloat16*,
    const MusaFusedElementwiseConfig&, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
