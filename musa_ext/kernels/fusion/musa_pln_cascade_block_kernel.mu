#include <musa_runtime.h>

#include "musa_pln_cascade_block_kernel.h"

namespace tensorflow {
namespace musa {

template <int kStaticSteps, bool kUseDynamicSteps>
__global__ void PlnCascadeBlockKernelImpl(
    const float* norm_out, PlnCascadeBlockStrides norm_out_st,
    PlnCascadeBlockGatePtrs gate_ptrs, PlnCascadeBlockMeta meta,
    const float* add_table, const float* bias_table, float* output,
    PlnCascadeBlockShape shape, int total_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  const int active_steps = kUseDynamicSteps ? meta.num_steps : kStaticSteps;
  if (active_steps <= 0) {
    output[idx] = norm_out[idx];
    return;
  }

  if (meta.use_fast_path) {
    const int channel_idx = idx % meta.table_width;
    const int batch_idx =
        (shape.rank > 1 && meta.output_inner_stride > 0)
            ? (idx / meta.output_inner_stride)
            : 0;

    float value = norm_out[idx];

#pragma unroll
    for (int step = 0; step < kStaticSteps; ++step) {
      if (kUseDynamicSteps && step >= active_steps) {
        break;
      }

      const bool* gate_ptr = gate_ptrs.values[step];
      if (gate_ptr == nullptr) {
        continue;
      }

      int gate_offset = 0;
      if (meta.gate_modes[step] == kPlnCascadeBlockGateModeBatchAligned) {
        gate_offset = batch_idx;
      }

      const bool gate = gate_ptr[gate_offset];
      const int table_offset = meta.table_base_offsets[step] + channel_idx;
      const float add_v = add_table[table_offset];
      const float bias_v = bias_table[table_offset];
      const float candidate = value * add_v + bias_v;

      const bool take_candidate = (gate == (meta.select_on_true[step] != 0));
      value = take_candidate ? candidate : value;
    }

    output[idx] = value;
    return;
  }

  int gate_offsets[kStaticSteps];
#pragma unroll
  for (int step = 0; step < kStaticSteps; ++step) {
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

#pragma unroll
    for (int step = 0; step < kStaticSteps; ++step) {
      if (!kUseDynamicSteps || step < active_steps) {
        gate_offsets[step] += coord * meta.gate_strides[step].values[dim];
      }
    }
  }

  if (channel_idx < 0 || channel_idx >= meta.table_width) {
    return;
  }

  float value = norm_out[norm_offset];

#pragma unroll
  for (int step = 0; step < kStaticSteps; ++step) {
    if (kUseDynamicSteps && step >= active_steps) {
      break;
    }

    const bool* gate_ptr = gate_ptrs.values[step];
    if (gate_ptr == nullptr) {
      continue;
    }

    const bool gate = gate_ptr[gate_offsets[step]];
    const int table_offset = meta.table_base_offsets[step] + channel_idx;
    const float add_v = add_table[table_offset];
    const float bias_v = bias_table[table_offset];
    const float candidate = value * add_v + bias_v;

    const bool take_candidate = (gate == (meta.select_on_true[step] != 0));
    value = take_candidate ? candidate : value;
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

#define LAUNCH_PLN_CASE(STEPS)                                               \
  case STEPS:                                                                   \
    PlnCascadeBlockKernelImpl<STEPS, false><<<grid_size, block_size, 0, stream>>>( \
        norm_out, norm_out_st, gate_ptrs, meta, add_table, bias_table, output, \
        shape, total_elements);                                                 \
    break

  switch (meta.num_steps) {
    LAUNCH_PLN_CASE(1);
    LAUNCH_PLN_CASE(2);
    LAUNCH_PLN_CASE(3);
    LAUNCH_PLN_CASE(4);
    LAUNCH_PLN_CASE(5);
    LAUNCH_PLN_CASE(6);
    LAUNCH_PLN_CASE(7);
    LAUNCH_PLN_CASE(8);
    LAUNCH_PLN_CASE(9);
    LAUNCH_PLN_CASE(10);
    LAUNCH_PLN_CASE(11);
    LAUNCH_PLN_CASE(12);
    LAUNCH_PLN_CASE(13);
    LAUNCH_PLN_CASE(14);
    LAUNCH_PLN_CASE(15);
    LAUNCH_PLN_CASE(16);
    default:
      PlnCascadeBlockKernelImpl<kPlnCascadeBlockMaxSteps, true>
          <<<grid_size, block_size, 0, stream>>>(
              norm_out, norm_out_st, gate_ptrs, meta, add_table, bias_table,
              output, shape, total_elements);
      break;
  }

#undef LAUNCH_PLN_CASE
}

}  // namespace musa
}  // namespace tensorflow
