#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaClipXlaOp : public XlaOpKernel {
 public:
  explicit MusaClipXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    BCast bcast_x_lo(BCast::FromShape(ctx->InputShape(0)),
                     BCast::FromShape(ctx->InputShape(1)));
    OP_REQUIRES(ctx, bcast_x_lo.IsValid(),
                errors::InvalidArgument("MusaClip x/lo shapes are not "
                                        "broadcast-compatible"));
    BCast bcast_all(bcast_x_lo.output_shape(),
                    BCast::FromShape(ctx->InputShape(2)));
    OP_REQUIRES(ctx, bcast_all.IsValid(),
                errors::InvalidArgument("MusaClip x/lo/hi shapes are not "
                                        "broadcast-compatible"));

    TensorShape out_shape = BCast::ToShape(bcast_all.output_shape());
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp lo = ctx->Input(1);
    xla::XlaOp hi = ctx->Input(2);
    BroadcastToShape(ctx, &x, out_shape);
    BroadcastToShape(ctx, &lo, out_shape);
    BroadcastToShape(ctx, &hi, out_shape);
    ctx->SetOutput(0, xla::Clamp(lo, x, hi));
  }
};

class MusaPReluXlaOp : public XlaOpKernel {
 public:
  explicit MusaPReluXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp x = ctx->Input(0);
    xla::XlaOp alpha = ctx->Input(1);
    BroadcastToShape(ctx, &alpha, ctx->InputShape(0));
    xla::XlaOp zero = xla::ZerosLike(x);
    ctx->SetOutput(0, xla::Max(zero, x) + alpha * xla::Min(zero, x));
  }
};

REGISTER_XLA_OP(Name("MusaClip"), MusaClipXlaOp);
REGISTER_XLA_OP(Name("MusaPRelu").TypeConstraint("T", kFloatTypes),
                MusaPReluXlaOp);

}  // namespace
}  // namespace tensorflow
