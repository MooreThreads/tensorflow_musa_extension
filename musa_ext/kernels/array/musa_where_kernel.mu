#include <musa_runtime.h>
#include <stdint.h>

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

template <int NDIM, typename TIndex>
struct StridesPack {
  TIndex v[NDIM];
};

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const StridesPack<NDIM, TIndex> strides,
    const TIndex* __restrict__ selected_indices, int64_t* __restrict__ output) {
  const TIndex i = static_cast<TIndex>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < output_rows) {
    TIndex index_value = selected_indices[i];
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      const TIndex stride = strides.v[c];
      *(output + NDIM * i + c) = index_value / stride;
      index_value %= stride;
    }
  }
}

template <int NDIM, typename TIndex>
musaError_t LaunchPropagateWhereIndicesKernel(const TIndex output_rows,
                                              const TIndex* strides_host,
                                              const TIndex* selected_indices,
                                              int64_t* output,
                                              musaStream_t stream) {
  if (output_rows <= static_cast<TIndex>(0)) {
    return musaSuccess;
  }

  StridesPack<NDIM, TIndex> pack;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    pack.v[i] = strides_host[i];
  }

  const int block_size = 256;
  const int grid_size =
      static_cast<int>((output_rows + block_size - 1) / block_size);
  PropagateWhereIndicesKernel<NDIM, TIndex>
      <<<grid_size, block_size, 0, stream>>>(output_rows, pack,
                         selected_indices, output);
  return musaGetLastError();
}

#define INSTANTIATE_PROPAGATE(NDIM, TINDEX)                                  \
  template musaError_t LaunchPropagateWhereIndicesKernel<NDIM, TINDEX>(      \
      const TINDEX output_rows, const TINDEX* strides_host,                   \
      const TINDEX* selected_indices, int64_t* output, musaStream_t stream)

INSTANTIATE_PROPAGATE(1, int32);
INSTANTIATE_PROPAGATE(2, int32);
INSTANTIATE_PROPAGATE(3, int32);
INSTANTIATE_PROPAGATE(4, int32);
INSTANTIATE_PROPAGATE(5, int32);
INSTANTIATE_PROPAGATE(6, int32);
INSTANTIATE_PROPAGATE(7, int32);
INSTANTIATE_PROPAGATE(8, int32);

INSTANTIATE_PROPAGATE(1, int64);
INSTANTIATE_PROPAGATE(2, int64);
INSTANTIATE_PROPAGATE(3, int64);
INSTANTIATE_PROPAGATE(4, int64);
INSTANTIATE_PROPAGATE(5, int64);
INSTANTIATE_PROPAGATE(6, int64);
INSTANTIATE_PROPAGATE(7, int64);
INSTANTIATE_PROPAGATE(8, int64);

#undef INSTANTIATE_PROPAGATE

}  // namespace musa
}  // namespace tensorflow