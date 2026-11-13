#ifndef TENSORFLOW_MUSA_KERNELS_EMPTY_MATMUL_H_
#define TENSORFLOW_MUSA_KERNELS_EMPTY_MATMUL_H_

#include "utils_op.h"

namespace tensorflow {
namespace musa {

// A zero contraction dimension yields zeros (or broadcast bias), even when
// the result is nonempty. Call only after validating all operand shapes.
inline Status InitializeEmptyMatMul(OpKernelContext* ctx, Tensor* output,
                                    const Tensor* bias = nullptr,
                                    bool relu = false) {
  if (output->NumElements() == 0) return OkStatus();
  auto stream = GetMusaStreamByCtx(ctx);
  if (musaMemsetAsync(output->data(), 0, output->TotalBytes(), stream) !=
      musaSuccess) {
    return errors::Internal("Failed to zero empty MatMul result");
  }
  if (bias == nullptr) return OkStatus();
  if (bias->NumElements() == 0 ||
      output->NumElements() % bias->NumElements() != 0) {
    return errors::InvalidArgument("Invalid bias for empty MatMul");
  }
  Tensor out_view, bias_view;
  if (!out_view.CopyFrom(
          *output, TensorShape({output->NumElements() / bias->NumElements(),
                                bias->NumElements()})) ||
      !bias_view.CopyFrom(*bias, TensorShape({bias->NumElements()}))) {
    return errors::Internal("Cannot reshape empty MatMul bias/result");
  }
  auto out_mt = CreateMTensor(out_view, mFormat::NCHW);
  auto bias_mt = CreateMTensor(bias_view, mFormat::NCHW);
  auto& handle = GetHandleByCtx(ctx);
  mBinary add;
  add.SetMode(BINARY_MODE::ADD);
  if (add.Run(handle, out_mt, out_mt, bias_mt) != mStatus::SUCCESS) {
    return errors::Internal("Failed to broadcast bias for empty MatMul");
  }
  if (relu) {
    ::musa::dnn::Unary activation;
    activation.SetMode(::musa::dnn::Unary::Mode::RELU);
    if (activation.Run(handle, out_mt, out_mt) != mStatus::SUCCESS) {
      return errors::Internal("Failed to activate empty MatMul result");
    }
  }
  return OkStatus();
}

}  // namespace musa
}  // namespace tensorflow
#endif
