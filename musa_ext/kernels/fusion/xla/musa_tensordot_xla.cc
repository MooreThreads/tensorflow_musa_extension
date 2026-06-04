#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaTensorDotXlaOp : public XlaOpKernel {
 public:
  explicit MusaTensorDotXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_a", &axes_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_b", &axes_b_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorDotDims dims;
    xla::XlaOp output;
    TensorDot(ctx, ctx->Input(0), ctx->InputShape(0), ctx->Input(1),
              ctx->InputShape(1), axes_a_, axes_b_, &dims, &output);
    ctx->SetOutput(0, output);
  }

 private:
  std::vector<int> axes_a_;
  std::vector<int> axes_b_;
};

class MusaTensorDotBiasXlaOp : public XlaOpKernel {
 public:
  explicit MusaTensorDotBiasXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_a", &axes_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_b", &axes_b_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorDotDims dims;
    xla::XlaOp output;
    TensorDot(ctx, ctx->Input(0), ctx->InputShape(0), ctx->Input(1),
              ctx->InputShape(1), axes_a_, axes_b_, &dims, &output);
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(dims.output_dims,
                                                      &out_shape));
    xla::XlaOp bias = ctx->Input(2);
    BroadcastToShape(ctx, &bias, out_shape);
    ctx->SetOutput(0, output + bias);
  }

 private:
  std::vector<int> axes_a_;
  std::vector<int> axes_b_;
};

REGISTER_XLA_OP(Name("MusaTensorDot").TypeConstraint("T", kFloatTypes),
                MusaTensorDotXlaOp);
REGISTER_XLA_OP(Name("MusaTensorDotBias").TypeConstraint("T", kFloatTypes),
                MusaTensorDotBiasXlaOp);

}  // namespace
}  // namespace tensorflow
