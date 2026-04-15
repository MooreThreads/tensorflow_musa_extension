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

__device__ __forceinline__ bool LoadElement(const bool* p) { return *p; }
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
    const void* const* ptrs, int input_index) {
  return reinterpret_cast<const T*>(ptrs[input_index]);
}

template <typename T>
__device__ __forceinline__ typename MusaFusedElementwiseAccumType<T>::type
LoadDataInputValue(MusaFusedElementwiseInlinePointers inputs, int input_index,
                   int offset) {
  const T* ptr = GetInputPointer<T>(inputs.data_ptrs, input_index);
  return static_cast<typename MusaFusedElementwiseAccumType<T>::type>(
      LoadElement(ptr + offset));
}

__device__ __forceinline__ bool LoadBoolInputValue(
    MusaFusedElementwiseInlinePointers inputs, int input_index, int offset) {
  const bool* ptr = GetInputPointer<bool>(inputs.bool_ptrs, input_index);
  return LoadElement(ptr + offset);
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
    case kOpcodePow:
      return pow(lhs, rhs);
    default:
      return lhs;
  }
}

template <typename AccT>
__device__ __forceinline__ AccT ApplySelect(bool cond, AccT then_val,
                                            AccT else_val) {
  return cond ? then_val : else_val;
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

  int data_input_offsets[kMusaFusedElementwiseMaxDataInputs] = {0};
  for (int input_idx = 0; input_idx < config.num_data_inputs; ++input_idx) {
    int offset = 0;
    for (int dim = 0; dim < config.rank; ++dim) {
      offset += coords[dim] * config.data_input_strides[input_idx][dim];
    }
    data_input_offsets[input_idx] = offset;
  }

  int bool_input_offsets[kMusaFusedElementwiseMaxBoolInputs] = {0};
  for (int input_idx = 0; input_idx < config.num_bool_inputs; ++input_idx) {
    int offset = 0;
    for (int dim = 0; dim < config.rank; ++dim) {
      offset += coords[dim] * config.bool_input_strides[input_idx][dim];
    }
    bool_input_offsets[input_idx] = offset;
  }

  using AccT = typename MusaFusedElementwiseAccumType<T>::type;
  AccT step_values[kMusaFusedElementwiseMaxSteps] = {AccT(0)};

  for (int step = 0; step < config.num_steps; ++step) {
    const int arg0_kind = config.step_arg_kind[step][0];
    const int arg0_input = config.step_arg_input[step][0];
    const AccT arg0 = arg0_kind == kOperandStep
                          ? step_values[arg0_input]
                          : LoadDataInputValue<T>(inputs, arg0_input,
                                                  data_input_offsets[arg0_input]);

    if (config.step_arity[step] == 1) {
      step_values[step] = ApplyUnary(config.step_opcode[step], arg0);
      continue;
    }

    if (config.step_arity[step] == 3) {
      const int cond_kind = config.step_arg_kind[step][0];
      const int cond_input = config.step_arg_input[step][0];
      const bool cond = cond_kind == kOperandStep
                            ? static_cast<bool>(step_values[cond_input])
                            : LoadBoolInputValue(inputs, cond_input,
                                                 bool_input_offsets[cond_input]);
      const int arg1_kind = config.step_arg_kind[step][1];
      const int arg1_input = config.step_arg_input[step][1];
      const AccT then_val =
          arg1_kind == kOperandStep
              ? step_values[arg1_input]
              : LoadDataInputValue<T>(inputs, arg1_input,
                                      data_input_offsets[arg1_input]);
      const int arg2_kind = config.step_arg_kind[step][2];
      const int arg2_input = config.step_arg_input[step][2];
      const AccT else_val =
          arg2_kind == kOperandStep
              ? step_values[arg2_input]
              : LoadDataInputValue<T>(inputs, arg2_input,
                                      data_input_offsets[arg2_input]);
      step_values[step] = ApplySelect(cond, then_val, else_val);
      continue;
    }

    const int arg1_kind = config.step_arg_kind[step][1];
    const int arg1_input = config.step_arg_input[step][1];
    const AccT arg1 = arg1_kind == kOperandStep
                          ? step_values[arg1_input]
                          : LoadDataInputValue<T>(inputs, arg1_input,
                                                  data_input_offsets[arg1_input]);
    step_values[step] = ApplyBinary(config.step_opcode[step], arg0, arg1);
  }

  StoreElement(output + idx, step_values[config.num_steps - 1]);
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
