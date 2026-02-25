#include "utils_op.h"

// #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// #include "tensorflow/core/kernels/reduction_ops_common_gpu.h"
// #endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace musa {

template <typename Device, typename T, int N>
struct StrideFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.stride(strides);
  }
};

template <typename Device, typename T, int N>
struct InflateFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.inflate(strides);
  }
};

}  // namespace musa
}  // namespace tensorflow