#ifndef TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_
#define TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

constexpr int kMusaFusedElementwiseMaxDims = 8;
constexpr int kMusaFusedElementwiseMaxInputs = 8;
constexpr int kMusaFusedElementwiseMaxSteps = 8;

enum MusaFusedElementwiseOperandKind : int {
  kOperandNone = -1,
  kOperandPrev = 0,
  kOperandInput = 1,
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
};

struct MusaFusedElementwiseInlinePointers {
  const void* ptrs[kMusaFusedElementwiseMaxInputs];
};

struct MusaFusedElementwiseConfig {
  int rank;
  int dims[kMusaFusedElementwiseMaxDims];
  int num_inputs;
  int num_steps;
  int input_strides[kMusaFusedElementwiseMaxInputs]
                   [kMusaFusedElementwiseMaxDims];
  int step_opcode[kMusaFusedElementwiseMaxSteps];
  int step_arity[kMusaFusedElementwiseMaxSteps];
  int step_arg_kind[kMusaFusedElementwiseMaxSteps][2];
  int step_arg_input[kMusaFusedElementwiseMaxSteps][2];
};

template <typename T>
void LaunchMusaFusedElementwiseKernel(
    MusaFusedElementwiseInlinePointers inputs, T* output,
    const MusaFusedElementwiseConfig& config, int total_elements,
    musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_FUSED_ELEMENTWISE_KERNEL_H_
