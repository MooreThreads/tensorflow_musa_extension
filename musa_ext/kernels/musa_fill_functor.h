/*
  Be advised:

  This file is implemented in aim to support the einsum operator.
  For now it only contains the SetZeroFunctor, which is used to set the output
  tensor to zero before accumulating the results of the einsum computation.

*/

#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace musa {

template <typename Device, typename T>
struct SetZeroFunctor {
  // Computes on device "d": out = out.setZero(),
  void operator()(const Device& d, typename TTypes<T>::Flat out) {
    out.device(d) = out.constant(T(0));
  }
};

}  // namespace musa
}  // namespace tensorflow