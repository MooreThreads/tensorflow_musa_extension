#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

xla::XlaOp Gelu(xla::XlaOp x, bool approximate) {
  xla::XlaOp half = xla::ScalarLike(x, 0.5);
  xla::XlaOp one = xla::ScalarLike(x, 1.0);
  if (approximate) {
    xla::XlaOp coeff = xla::ScalarLike(x, std::sqrt(2.0 / M_PI));
    xla::XlaOp cubic_coeff = xla::ScalarLike(x, 0.044715);
    xla::XlaOp x_cubed = x * x * x;
    return half * x *
           (one + xla::Tanh(coeff * (x + cubic_coeff * x_cubed)));
  }

  xla::XlaOp rsqrt_two = xla::ScalarLike(x, 1.0 / std::sqrt(2.0));
  return half * x * (one + xla::Erf(x * rsqrt_two));
}

xla::XlaOp GeluGrad(xla::XlaOp dy, xla::XlaOp x, bool approximate) {
  xla::XlaOp half = xla::ScalarLike(x, 0.5);
  xla::XlaOp one = xla::ScalarLike(x, 1.0);
  if (approximate) {
    xla::XlaOp coeff = xla::ScalarLike(x, std::sqrt(2.0 / M_PI));
    xla::XlaOp cubic_coeff = xla::ScalarLike(x, 0.044715);
    xla::XlaOp three = xla::ScalarLike(x, 3.0);
    xla::XlaOp x_sq = x * x;
    xla::XlaOp tanh_arg = coeff * (x + cubic_coeff * x * x_sq);
    xla::XlaOp tanh_val = xla::Tanh(tanh_arg);
    xla::XlaOp du_dx = coeff * (one + three * cubic_coeff * x_sq);
    xla::XlaOp grad =
        half * (one + tanh_val) + half * x * (one - tanh_val * tanh_val) * du_dx;
    return dy * grad;
  }

  xla::XlaOp rsqrt_two = xla::ScalarLike(x, 1.0 / std::sqrt(2.0));
  xla::XlaOp inv_sqrt_2pi = xla::ScalarLike(x, 1.0 / std::sqrt(2.0 * M_PI));
  xla::XlaOp cdf = half * (one + xla::Erf(x * rsqrt_two));
  xla::XlaOp pdf = xla::Exp(xla::ScalarLike(x, -0.5) * x * x) * inv_sqrt_2pi;
  return dy * (cdf + x * pdf);
}

class MusaGeluXlaOp : public XlaOpKernel {
 public:
  explicit MusaGeluXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, Gelu(ctx->Input(0), approximate_));
  }

 private:
  bool approximate_;
};

class MusaGeluGradXlaOp : public XlaOpKernel {
 public:
  explicit MusaGeluGradXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->InputShape(0) == ctx->InputShape(1),
        errors::InvalidArgument("MusaGeluGrad requires dy and x same shape"));
    ctx->SetOutput(0, GeluGrad(ctx->Input(0), ctx->Input(1), approximate_));
  }

 private:
  bool approximate_;
};

REGISTER_XLA_OP(Name("MusaGelu").TypeConstraint("T", kFloatTypes),
                MusaGeluXlaOp);
REGISTER_XLA_OP(Name("MusaGeluGrad").TypeConstraint("T", kFloatTypes),
                MusaGeluGradXlaOp);

}  // namespace
}  // namespace tensorflow
