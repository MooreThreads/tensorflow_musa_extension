#include <musa_runtime.h>

#include "musa_pln_cascade_block_kernel.h"

namespace tensorflow {
namespace musa {

__global__ void PlnCascadeBlockKernel(
    const float* norm_out, PlnCascadeBlockStrides norm_out_st,
    PlnCascadeBlockGatePtrs gate_ptrs, PlnCascadeBlockMeta meta,
    const float* add_table, const float* bias_table, float* output,
    PlnCascadeBlockShape shape, int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  int gate_offsets[kPlnCascadeBlockMaxSteps];
  for (int step = 0; step < kPlnCascadeBlockMaxSteps; ++step) {
    gate_offsets[step] = 0;
  }

  int remaining = idx;
  int norm_offset = 0;
  int channel_idx = 0;

  for (int dim = shape.rank - 1; dim >= 0; --dim) {
    const int coord = remaining % shape.dims[dim];
    remaining /= shape.dims[dim];

    norm_offset += coord * norm_out_st.values[dim];
    if (dim == shape.rank - 1) {
      channel_idx = coord;
    }

    for (int step = 0; step < meta.num_steps; ++step) {
      gate_offsets[step] += coord * meta.gate_strides[step].values[dim];
    }
  }

  if (channel_idx < 0 || channel_idx >= meta.table_width) {
    return;
  }

  float value = norm_out[norm_offset];

  for (int step = 0; step < meta.num_steps; ++step) {
    const int table_index = meta.table_indices[step];
    if (table_index < 0 || table_index >= meta.table_rows) {
      continue;
    }

    const bool* gate_ptr = gate_ptrs.values[step];
    if (gate_ptr == nullptr) {
      continue;
    }

    const bool gate = gate_ptr[gate_offsets[step]];
    const int table_offset = table_index * meta.table_width + channel_idx;
    const float add_v = add_table[table_offset];
    const float bias_v = bias_table[table_offset];
    const float candidate = value * add_v + bias_v;

    if (meta.select_on_true[step] != 0) {
      value = gate ? candidate : value;
    } else {
      value = gate ? value : candidate;
    }
  }

  output[idx] = value;
}

void LaunchPlnCascadeBlockKernel(const float* norm_out,
                                 PlnCascadeBlockStrides norm_out_st,
                                 PlnCascadeBlockGatePtrs gate_ptrs,
                                 PlnCascadeBlockMeta meta,
                                 const float* add_table,
                                 const float* bias_table, float* output,
                                 PlnCascadeBlockShape shape,
                                 int total_elements, musaStream_t stream) {
  if (total_elements <= 0) {
    return;
  }

  const int block_size = 256;
  const int grid_size = (total_elements + block_size - 1) / block_size;
  PlnCascadeBlockKernel<<<grid_size, block_size, 0, stream>>>(
      norm_out, norm_out_st, gate_ptrs, meta, add_table, bias_table, output,
      shape, total_elements);
}

}  // namespace musa
}  // namespace tensorflow
