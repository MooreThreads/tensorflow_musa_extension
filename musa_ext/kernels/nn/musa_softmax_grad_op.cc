#include <musa_runtime.h>

#include <cstdint>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {
namespace {

extern "C" void LaunchMusaSmallLastDimSoftmaxGradFloat(
    const float* softmax, const float* dy, float* dx, int64_t outer_size,
    int last_dim, musaStream_t stream);
extern "C" void LaunchMusaSmallLastDimSoftmaxGradBFloat16(
    const tensorflow::bfloat16* softmax, const tensorflow::bfloat16* dy,
    tensorflow::bfloat16* dx, int64_t outer_size, int last_dim,
    musaStream_t stream);

class MusaSoftmaxGradOp : public MusaOpKernel {
 public:
  explicit MusaSoftmaxGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& softmax = ctx->input(0);
    const Tensor& dy = ctx->input(1);
    OP_REQUIRES(ctx, softmax.dtype() == dy.dtype() &&
                         (softmax.dtype() == DT_FLOAT ||
                          softmax.dtype() == DT_BFLOAT16),
                errors::InvalidArgument(
                    "MusaSoftmaxGrad expects float/bfloat16"));
    OP_REQUIRES(ctx, softmax.shape() == dy.shape(),
                errors::InvalidArgument(
                    "MusaSoftmaxGrad input shape mismatch: ",
                    softmax.shape().DebugString(), " vs ",
                    dy.shape().DebugString()));
    OP_REQUIRES(ctx, softmax.dims() >= 1,
                errors::InvalidArgument("MusaSoftmaxGrad expects rank >= 1"));

    const int64_t last_dim_i64 = softmax.dim_size(softmax.dims() - 1);
    OP_REQUIRES(ctx, last_dim_i64 > 0 && last_dim_i64 <= 64,
                errors::InvalidArgument(
                    "MusaSoftmaxGrad supports last_dim in (0, 64], got ",
                    last_dim_i64));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, softmax.shape(), &dx));
    if (dx->NumElements() == 0) return;

    const int64_t outer_size = dx->NumElements() / last_dim_i64;
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    if (softmax.dtype() == DT_FLOAT) {
      LaunchMusaSmallLastDimSoftmaxGradFloat(
          softmax.flat<float>().data(), dy.flat<float>().data(),
          dx->flat<float>().data(), outer_size, static_cast<int>(last_dim_i64),
          stream);
    } else {
      LaunchMusaSmallLastDimSoftmaxGradBFloat16(
          softmax.flat<bfloat16>().data(), dy.flat<bfloat16>().data(),
          dx->flat<bfloat16>().data(), outer_size,
          static_cast<int>(last_dim_i64), stream);
    }
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaSoftmaxGrad kernel failed: ",
                                 musaGetErrorString(status)));
  }
};

#define REGISTER_MUSA_SOFTMAX_GRAD(TYPE)                              \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MusaSoftmaxGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSoftmaxGradOp);

REGISTER_MUSA_SOFTMAX_GRAD(float);
REGISTER_MUSA_SOFTMAX_GRAD(bfloat16);

#undef REGISTER_MUSA_SOFTMAX_GRAD

}  // namespace
}  // namespace musa

REGISTER_OP("MusaSoftmaxGrad")
    .Input("softmax: T")
    .Input("dy: T")
    .Output("dx: T")
    .Attr("T: {float, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

}  // namespace tensorflow
