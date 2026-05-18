#ifndef TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_BLOCK_KERNEL_H_
#define TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_BLOCK_KERNEL_H_

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

constexpr int kPlnCascadeBlockMaxDims = 8;
constexpr int kPlnCascadeBlockMaxSteps = 16;

enum PlnCascadeBlockGateMode {
  kPlnCascadeBlockGateModeGeneric = 0,
  kPlnCascadeBlockGateModeScalar = 1,
  kPlnCascadeBlockGateModeBatchAligned = 2,
};

struct PlnCascadeBlockShape {
  int rank;
  int dims[kPlnCascadeBlockMaxDims];
};

struct PlnCascadeBlockStrides {
  int values[kPlnCascadeBlockMaxDims];
};

struct PlnCascadeBlockGatePtrs {
  const bool* values[kPlnCascadeBlockMaxSteps];
};

struct PlnCascadeBlockMeta {
  int num_steps;
  int table_rows;
  int table_width;
  int norm_is_contiguous;
  int output_inner_stride;
  int use_fast_path;
  int table_indices[kPlnCascadeBlockMaxSteps];
  int table_base_offsets[kPlnCascadeBlockMaxSteps];
  int select_on_true[kPlnCascadeBlockMaxSteps];
  int gate_modes[kPlnCascadeBlockMaxSteps];
  PlnCascadeBlockStrides gate_strides[kPlnCascadeBlockMaxSteps];
};

void LaunchPlnCascadeBlockKernel(
    const float* norm_out, PlnCascadeBlockStrides norm_out_st,
    PlnCascadeBlockGatePtrs gate_ptrs, PlnCascadeBlockMeta meta,
    const float* add_table, const float* bias_table, float* output,
    PlnCascadeBlockShape shape, int total_elements, musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_BLOCK_KERNEL_H_
