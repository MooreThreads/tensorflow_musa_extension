#ifndef TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_KERNEL_H_
#define TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_KERNEL_H_

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {

constexpr int kPlnCascadeMaxDims = 8;

struct PlnCascadeShape {
  int rank;
  int dims[kPlnCascadeMaxDims];
};

struct PlnCascadeStrides {
  int values[kPlnCascadeMaxDims];
};

void LaunchPlnCascadeDirectKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_value,
    PlnCascadeStrides add_value_st, const float* bias_value,
    PlnCascadeStrides bias_value_st, float* output, PlnCascadeShape shape,
    int total_elements, bool select_on_true, musaStream_t stream);

void LaunchPlnCascadeTableKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_table, const float* bias_table,
    int table_rows, int table_width, int table_index, float* output,
    PlnCascadeShape shape, int total_elements, bool select_on_true,
    musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_PLN_CASCADE_KERNEL_H_
