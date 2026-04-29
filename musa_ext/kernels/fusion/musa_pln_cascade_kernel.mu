#include <musa_runtime.h>

#include "musa_pln_cascade_kernel.h"

namespace tensorflow {
namespace musa {

__global__ void PlnCascadeDirectKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_value,
    PlnCascadeStrides add_value_st, const float* bias_value,
    PlnCascadeStrides bias_value_st, float* output, PlnCascadeShape shape,
    int total_elements, bool select_on_true) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  int remaining = idx;
  int norm_offset = 0;
  int adpos_offset = 0;
  int add_offset = 0;
  int bias_offset = 0;

  for (int dim = shape.rank - 1; dim >= 0; --dim) {
    const int coord = remaining % shape.dims[dim];
    remaining /= shape.dims[dim];

    norm_offset += coord * norm_out_st.values[dim];
    adpos_offset += coord * adpos_st.values[dim];
    add_offset += coord * add_value_st.values[dim];
    bias_offset += coord * bias_value_st.values[dim];
  }

  const float norm_v = norm_out[norm_offset];
  const float add_v = add_value[add_offset];
  const float bias_v = bias_value[bias_offset];
  const bool adpos_v = adpos[adpos_offset];

  const float candidate = norm_v * add_v + bias_v;
  output[idx] = select_on_true ? (adpos_v ? candidate : norm_v)
                               : (adpos_v ? norm_v : candidate);
}

__global__ void PlnCascadeTableKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_table, const float* bias_table,
    int table_rows, int table_width, int table_index, float* output,
    PlnCascadeShape shape, int total_elements, bool select_on_true) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  int remaining = idx;
  int norm_offset = 0;
  int adpos_offset = 0;
  int channel_idx = 0;

  for (int dim = shape.rank - 1; dim >= 0; --dim) {
    const int coord = remaining % shape.dims[dim];
    remaining /= shape.dims[dim];

    norm_offset += coord * norm_out_st.values[dim];
    adpos_offset += coord * adpos_st.values[dim];
    if (dim == shape.rank - 1) {
      channel_idx = coord;
    }
  }

  if (table_index < 0 || table_index >= table_rows || channel_idx < 0 ||
      channel_idx >= table_width) {
    return;
  }

  const int table_offset = table_index * table_width + channel_idx;

  const float norm_v = norm_out[norm_offset];
  const float add_v = add_table[table_offset];
  const float bias_v = bias_table[table_offset];
  const bool adpos_v = adpos[adpos_offset];

  const float candidate = norm_v * add_v + bias_v;
  output[idx] = select_on_true ? (adpos_v ? candidate : norm_v)
                               : (adpos_v ? norm_v : candidate);
}

void LaunchPlnCascadeDirectKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_value,
    PlnCascadeStrides add_value_st, const float* bias_value,
    PlnCascadeStrides bias_value_st, float* output, PlnCascadeShape shape,
    int total_elements, bool select_on_true, musaStream_t stream) {
  if (total_elements <= 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (total_elements + block_size - 1) / block_size;
  PlnCascadeDirectKernel<<<grid_size, block_size, 0, stream>>>(
      norm_out, norm_out_st, adpos, adpos_st, add_value, add_value_st,
      bias_value, bias_value_st, output, shape, total_elements, select_on_true);
}

void LaunchPlnCascadeTableKernel(
    const float* norm_out, PlnCascadeStrides norm_out_st, const bool* adpos,
    PlnCascadeStrides adpos_st, const float* add_table, const float* bias_table,
    int table_rows, int table_width, int table_index, float* output,
    PlnCascadeShape shape, int total_elements, bool select_on_true,
    musaStream_t stream) {
  if (total_elements <= 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (total_elements + block_size - 1) / block_size;
  PlnCascadeTableKernel<<<grid_size, block_size, 0, stream>>>(
      norm_out, norm_out_st, adpos, adpos_st, add_table, bias_table, table_rows,
      table_width, table_index, output, shape, total_elements, select_on_true);
}

}  // namespace musa
}  // namespace tensorflow
