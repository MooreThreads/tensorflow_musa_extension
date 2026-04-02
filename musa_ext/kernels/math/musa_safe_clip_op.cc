#include <algorithm>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchSafeClip(const musaStream_t stream, const T* x_ptr, const int n,
                    const T* lo_ptr, const T* hi_ptr, T* y_ptr);

template <typename T>
class MusaSafeClipOp : public MusaOpKernel {
 public:
  explicit MusaSafeClipOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_x = ctx->input(0);
    const Tensor& input_lo = ctx->input(1);
    const Tensor& input_hi = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input_lo.shape()),
                errors::InvalidArgument("MusaSafeClip: lo must be scalar, "
                                        "received shape: ",
                                        input_lo.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input_hi.shape()),
                errors::InvalidArgument("MusaSafeClip: hi must be scalar, "
                                        "received shape: ",
                                        input_hi.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_x.shape(), &output));
    if (output->NumElements() == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchSafeClip<T>(stream, input_x.flat<T>().data(), input_x.NumElements(),
                      input_lo.flat<T>().data(), input_hi.flat<T>().data(),
                      output->flat<T>().data());
  }
};

#define REGISTER_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MusaSafeClip").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaSafeClipOp<type>)

TF_CALL_float(REGISTER_KERNELS);
TF_CALL_double(REGISTER_KERNELS);
TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_int64(REGISTER_KERNELS);
TF_CALL_half(REGISTER_KERNELS);
TF_CALL_bfloat16(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace musa

REGISTER_OP("MusaSafeClip")
    .Input("x: T")
    .Input("lo: T")
    .Input("hi: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16, double, int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
