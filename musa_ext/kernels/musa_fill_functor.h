/*
  Be advised:

  This file is implemented in aim to support the einsum operator.
  For now it only contains the SetZeroFunctor, which is used to set the output
  tensor to zero before accumulating the results of the einsum computation.

*/

#include <type_traits>

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
Status MusaFillCall(mTensor* out_mt, T value, OpKernelContext* context) {
  mFill op;
  mHandle& h = GetHandleByCtx(context);

  if (std::is_integral<T>::value) {
    if (mStatus::SUCCESS != op.SetValue(static_cast<int64_t>(value))) {
      return errors::Internal("mtdnn set value (int) error!");
    }
  } else if (std::is_floating_point<T>::value ||
             std::is_same<T, Eigen::half>::value ||
             std::is_same<T, Eigen::bfloat16>::value) {
    if (mStatus::SUCCESS != op.SetValue(static_cast<double>(value))) {
      return errors::Internal("mtdnn set value (float) error!");
    }
  } else {
    return errors::Unimplemented("Data type not supported in MTGPU Fill.");
  }

  if (mStatus::SUCCESS != op.Run(h, *out_mt)) {
    return errors::Internal("mtdnn run op error!");
  }

  return Status::OK();
}

struct SetZeroFunctor {
  // Computes on device "d": out = out.setZero(),
  template <typename T>
  static Status Compute(OpKernelContext* ctx, mTensor* out_mt) {
    return MusaFillCall<T>(out_mt, T(0), ctx);
  }
};

}  // namespace musa
}  // namespace tensorflow