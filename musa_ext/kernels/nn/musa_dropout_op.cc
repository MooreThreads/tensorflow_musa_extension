#include <mudnn.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

// ----------------------------------------------------------------------------
// Forward: MusaDropoutOp
// Inputs : x (T)
// Outputs: y (T), mask (uint8)
// Attrs  : rate (float, drop probability), seed (int64), offset (int64)
// ----------------------------------------------------------------------------
template <typename T>
class MusaDropoutOp : public MusaOpKernel {
 public:
  explicit MusaDropoutOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rate", &rate_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offset", &offset_));

    OP_REQUIRES(
        ctx, rate_ >= 0.0f && rate_ < 1.0f,
        errors::InvalidArgument("Dropout rate must be in [0, 1), got ", rate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    // mask is bool, same shape as x
    Tensor* mask = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x.shape(), &mask));

    if (x.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_x = CreateMTensor(x, format_);
    mTensor mt_y = CreateMTensor(*y, format_);
    mTensor mt_mask = CreateMTensor(*mask, format_);

    mDropout dropout;

    MTOP_CHECK_OK(dropout.SetP(static_cast<double>(rate_)), "Dropout SetP",
                  ctx);
    // Inverted dropout scaling: 1 / (1 - rate)
    const double scale =
        (rate_ < 1.0f) ? (1.0 / (1.0 - static_cast<double>(rate_))) : 0.0;
    MTOP_CHECK_OK(dropout.SetScale(scale), "Dropout SetScale", ctx);
    MTOP_CHECK_OK(dropout.SetSeed(static_cast<uint64_t>(seed_)),
                  "Dropout SetSeed", ctx);
    MTOP_CHECK_OK(dropout.SetOffset(static_cast<uint64_t>(offset_)),
                  "Dropout SetOffset", ctx);

    MTOP_CHECK_OK_RUN(dropout.RunDropout(handle, mt_y, mt_x, mt_mask),
                      "Dropout RunDropout", ctx);
  }

 private:
  float rate_;
  int64 seed_;
  int64 offset_;
};

// ----------------------------------------------------------------------------
// Backward: MusaDropoutGradOp
// Inputs : grad (T), mask (uint8)
// Outputs: grad_input (T)
// Attrs  : rate (float)
// ----------------------------------------------------------------------------
template <typename T>
class MusaDropoutGradOp : public MusaOpKernel {
 public:
  explicit MusaDropoutGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rate", &rate_));

    OP_REQUIRES(
        ctx, rate_ >= 0.0f && rate_ < 1.0f,
        errors::InvalidArgument("Dropout rate must be in [0, 1), got ", rate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad = ctx->input(0);
    const Tensor& mask = ctx->input(1);

    OP_REQUIRES(ctx, grad.shape() == mask.shape(),
                errors::InvalidArgument(
                    "grad and mask must have the same shape. grad: ",
                    grad.shape().DebugString(),
                    ", mask: ", mask.shape().DebugString()));

    Tensor* grad_input = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad.shape(), &grad_input));

    if (grad.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_grad = CreateMTensor(grad, format_);
    mTensor mt_mask = CreateMTensor(mask, format_);
    mTensor mt_grad_input = CreateMTensor(*grad_input, format_);

    mDropout dropout;

    MTOP_CHECK_OK(dropout.SetP(static_cast<double>(rate_)), "DropoutGrad SetP",
                  ctx);
    const double scale =
        (rate_ < 1.0f) ? (1.0 / (1.0 - static_cast<double>(rate_))) : 0.0;
    MTOP_CHECK_OK(dropout.SetScale(scale), "DropoutGrad SetScale", ctx);

    MTOP_CHECK_OK_RUN(
        dropout.RunDropoutBwd(handle, mt_grad_input, mt_grad, mt_mask),
        "Dropout RunDropoutBwd", ctx);
  }

 private:
  float rate_;
};

// ----------------------------------------------------------------------------
// Kernel registrations
// ----------------------------------------------------------------------------
#define REGISTER_MUSA_DROPOUT(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaDropout").Device("MUSA").TypeConstraint<TYPE>("T"),     \
      MusaDropoutOp<TYPE>);                                             \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaDropoutGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaDropoutGradOp<TYPE>);

REGISTER_MUSA_DROPOUT(float);
REGISTER_MUSA_DROPOUT(Eigen::half);
REGISTER_MUSA_DROPOUT(bfloat16);

#undef REGISTER_MUSA_DROPOUT

}  // namespace musa

// ----------------------------------------------------------------------------
// Op registrations
// ----------------------------------------------------------------------------
REGISTER_OP("MusaDropout")
    .Input("x: T")
    .Output("y: T")
    .Output("mask: bool")
    .Attr("T: {float, half, bfloat16}")
    .Attr("rate: float = 0.5")
    .Attr("seed: int = 0")
    .Attr("offset: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return Status::OK();
    });

REGISTER_OP("MusaDropoutGrad")
    .Input("grad: T")
    .Input("mask: bool")
    .Output("grad_input: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("rate: float = 0.5")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
