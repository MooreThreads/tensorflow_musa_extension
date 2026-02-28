#include "musa_stride_inflate_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, int N>
struct StrideFunctor {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    const Eigen::DenseIndex num_elements = output.size();
    if (num_elements == 0) return;

    DimSizeArray dims = {};
    DimSizeArray stride_dims = {};
    for (int i = 0; i < N; ++i) {
      dims.value[i] = static_cast<int64_t>(output.dimension(i));
      stride_dims.value[i] = static_cast<int64_t>(strides[i]);
    }

    auto stream = GetMusaStreamByCtx(ctx);
    MusaStrideKernelLauncher<T>(stream, static_cast<int64_t>(num_elements),
                                input.data(), output.data(), dims, stride_dims,
                                N);
  }
};

template <typename T, int N>
struct InflateFunctor {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    const Eigen::DenseIndex input_elements = input.size();
    const Eigen::DenseIndex output_elements = output.size();
    const int64_t output_count = static_cast<int64_t>(output_elements);

    auto stream = GetMusaStreamByCtx(ctx);
    if (output_count > 0) {
      const uint64_t bytes = output_count * sizeof(T);
      musaMemsetAsync(output.data(), 0, bytes, stream);
    }
    if (input_elements == 0 || output_elements == 0) return;

    DimSizeArray input_dims = {};
    DimSizeArray stride_dims = {};
    for (int i = 0; i < N; ++i) {
      input_dims.value[i] = static_cast<int64_t>(input.dimension(i));
      stride_dims.value[i] = static_cast<int64_t>(strides[i]);
    }

    MusaInflateKernelLauncher<T>(stream, static_cast<int64_t>(input_elements),
                                 input.data(), output.data(), input_dims,
                                 stride_dims, N, output_count);
  }
};

}  // namespace musa
}  // namespace tensorflow