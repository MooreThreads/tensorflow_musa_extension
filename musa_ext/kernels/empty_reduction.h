#ifndef TENSORFLOW_MUSA_KERNELS_EMPTY_REDUCTION_H_
#define TENSORFLOW_MUSA_KERNELS_EMPTY_REDUCTION_H_

#include <limits>
#include <vector>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace musa {

enum class EmptyReduction { kSum, kProd, kMean, kMin, kMax, kAll };

// Empty axes mean identity, not reduce-all. Empty inputs can have nonempty
// outputs; validate axes and construct the reduced shape before filling the
// operation's identity. No empty descriptor is passed to muDNN.
template <typename T>
Status HandleEmptyReduction(OpKernelContext* ctx, bool keep_dims,
                            EmptyReduction mode, bool* handled) {
  const Tensor& input = ctx->input(0);
  const Tensor& axes = ctx->input(1);  // Registered as HostMemory.
  *handled = input.NumElements() == 0 || axes.NumElements() == 0;
  if (!*handled) return OkStatus();
  if (axes.NumElements() == 0) {
    ctx->set_output(0, input);
    return OkStatus();
  }
  std::vector<bool> reduced(input.dims(), false);
  for (int64_t i = 0; i < axes.NumElements(); ++i) {
    int64_t axis = axes.dtype() == DT_INT32 ? axes.flat<int32>()(i)
                                            : axes.flat<int64_t>()(i);
    if (axis < 0) axis += input.dims();
    if (axis < 0 || axis >= input.dims()) {
      return errors::InvalidArgument("Invalid reduction dimension ", axis,
                                     " for input with ", input.dims(),
                                     " dimensions");
    }
    reduced[axis] = true;
  }
  TensorShape shape;
  for (int d = 0; d < input.dims(); ++d) {
    if (!reduced[d])
      shape.AddDim(input.dim_size(d));
    else if (keep_dims)
      shape.AddDim(1);
  }
  Tensor* output = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_output(0, shape, &output));
  if (output->NumElements() == 0) return OkStatus();

  T value = T(0);
  switch (mode) {
    case EmptyReduction::kProd:
    case EmptyReduction::kAll:
      value = T(1);
      break;
    case EmptyReduction::kMean:
      value = std::numeric_limits<T>::quiet_NaN();
      break;
    case EmptyReduction::kMin:
      value = std::numeric_limits<T>::has_infinity
                  ? std::numeric_limits<T>::infinity()
                  : std::numeric_limits<T>::max();
      break;
    case EmptyReduction::kMax:
      value = std::numeric_limits<T>::has_infinity
                  ? T(-std::numeric_limits<T>::infinity())
                  : std::numeric_limits<T>::lowest();
      break;
    case EmptyReduction::kSum:
      break;
  }
  // The synchronous copy keeps the host buffer alive until it is consumed.
  Tensor host(input.dtype(), shape);
  host.flat<T>().setConstant(value);
  if (MusaMemcpyH2D(output->data(), host.data(), host.TotalBytes()) !=
      mStatus::SUCCESS) {
    return errors::Internal("Failed to initialize empty reduction output");
  }
  return OkStatus();
}

}  // namespace musa
}  // namespace tensorflow
#endif  // TENSORFLOW_MUSA_KERNELS_EMPTY_REDUCTION_H_
