#ifndef TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_
#define TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

constexpr int kMusaFusedElementwiseMaxDims = 8;
constexpr int kMusaFusedElementwiseMaxDataInputs = 8;
constexpr int kMusaFusedElementwiseMaxBoolInputs = 4;
constexpr int kMusaFusedElementwiseMaxSteps = 16;
constexpr int kMusaFusedElementwiseMaxArity = 3;

enum MusaFusedElementwiseOperandKind : int {
  kOperandNone = -1,
  kOperandDataInput = 0,
  kOperandBoolInput = 1,
  kOperandStep = 2,
};

enum MusaFusedElementwiseOpcode : int {
  kOpcodeAdd = 1,
  kOpcodeSub = 2,
  kOpcodeMul = 3,
  kOpcodeRealDiv = 4,
  kOpcodeExp = 5,
  kOpcodeLog = 6,
  kOpcodeRsqrt = 7,
  kOpcodeRelu = 8,
  kOpcodeTanh = 9,
  kOpcodeSigmoid = 10,
  kOpcodeMaximum = 11,
  kOpcodeMinimum = 12,
  kOpcodeNeg = 13,
  kOpcodePow = 14,
  kOpcodeSelect = 15,
};

struct MusaFusedElementwiseInlinePointers {
  const void* data_ptrs[kMusaFusedElementwiseMaxDataInputs];
  const void* bool_ptrs[kMusaFusedElementwiseMaxBoolInputs];
};

struct MusaFusedElementwiseConfig {
  int rank;
  int dims[kMusaFusedElementwiseMaxDims];
  int num_data_inputs;
  int num_bool_inputs;
  int num_steps;
  int data_input_strides[kMusaFusedElementwiseMaxDataInputs]
                        [kMusaFusedElementwiseMaxDims];
  int bool_input_strides[kMusaFusedElementwiseMaxBoolInputs]
                        [kMusaFusedElementwiseMaxDims];
  int step_opcode[kMusaFusedElementwiseMaxSteps];
  int step_arity[kMusaFusedElementwiseMaxSteps];
  int step_arg_kind[kMusaFusedElementwiseMaxSteps]
                   [kMusaFusedElementwiseMaxArity];
  int step_arg_input[kMusaFusedElementwiseMaxSteps]
                    [kMusaFusedElementwiseMaxArity];
};

template <typename T>
void LaunchMusaFusedElementwiseKernel(
    MusaFusedElementwiseInlinePointers inputs, T* output,
    const MusaFusedElementwiseConfig& config, int total_elements,
    musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_
