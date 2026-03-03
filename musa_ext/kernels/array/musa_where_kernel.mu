#include <musa_runtime.h>
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,
    int64* __restrict__ output) {
  const TIndex i = static_cast<TIndex>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < output_rows) {
    TIndex index_value = ldg(output + NDIM * i);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];
      index_value %= strides[c];
    }
  }
}

}  // namespace musa
}  // namespace tensorflow