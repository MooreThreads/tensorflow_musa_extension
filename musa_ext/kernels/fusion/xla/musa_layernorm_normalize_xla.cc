#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaLayerNormXlaOp : public XlaOpKernel {
 public:
  explicit MusaLayerNormXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileNormalize(ctx, /*use_affine=*/true, epsilon_,
                     /*max_std=*/std::numeric_limits<float>::infinity(),
                     /*layernorm_eps_inside_sqrt=*/true);
  }

 private:
  float epsilon_;
};

class MusaLayerNormGradXlaOp : public XlaOpKernel {
 public:
  explicit MusaLayerNormGradXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape dy_shape = ctx->InputShape(0);
    const TensorShape x_shape = ctx->InputShape(1);
    const TensorShape gamma_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, dy_shape == x_shape && x_shape.dims() >= 1,
                errors::InvalidArgument(
                    "MusaLayerNormGrad requires dy/x same rank >= 1 shape"));
    const int64_t rank = x_shape.dims();
    const int64_t last_dim = x_shape.dim_size(rank - 1);
    const int64_t num_rows = x_shape.num_elements() / last_dim;
    OP_REQUIRES(ctx, gamma_shape.num_elements() == last_dim &&
                         ctx->InputShape(3).num_elements() == last_dim,
                errors::InvalidArgument(
                    "MusaLayerNormGrad gamma/beta size must match last dim"));

    const DataType out_type = ctx->input_type(0);
    const DataType compute_type =
        out_type == DT_DOUBLE ? DT_DOUBLE
                              : (out_type == DT_FLOAT ? DT_DOUBLE : DT_FLOAT);
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp dy = xla::Reshape(ConvertTo(ctx->Input(0), compute_type),
                                 {num_rows, last_dim});
    xla::XlaOp x = xla::Reshape(ConvertTo(ctx->Input(1), compute_type),
                                {num_rows, last_dim});
    xla::XlaOp gamma = ConvertTo(ctx->Input(2), compute_type);
    xla::XlaOp divisor = XlaHelpers::FloatLiteral(b, compute_type, last_dim);

    xla::XlaOp mean_col =
        RowWiseSum2D(ctx, x, compute_type, num_rows, last_dim) / divisor;
    xla::XlaOp centered = x - BroadcastCols(mean_col, num_rows, last_dim);
    xla::XlaOp variance_col =
        RowWiseSum2D(ctx, centered * centered, compute_type, num_rows,
                     last_dim) /
        divisor;
    xla::XlaOp inv_std_col =
        xla::Rsqrt(variance_col +
                   XlaHelpers::FloatLiteral(b, compute_type, epsilon_));
    xla::XlaOp inv_std = BroadcastCols(inv_std_col, num_rows, last_dim);
    xla::XlaOp x_hat = centered * inv_std;

    xla::XlaOp gamma_b = xla::BroadcastInDim(gamma, {num_rows, last_dim}, {1});
    xla::XlaOp dxhat = dy * gamma_b;
    xla::XlaOp mean_dxhat =
        RowWiseSum2D(ctx, dxhat, compute_type, num_rows, last_dim) / divisor;
    xla::XlaOp mean_dxhat_xhat =
        RowWiseSum2D(ctx, dxhat * x_hat, compute_type, num_rows, last_dim) /
        divisor;
    xla::XlaOp dx_2d = inv_std *
                       (dxhat - BroadcastCols(mean_dxhat, num_rows, last_dim) -
                        x_hat *
                            BroadcastCols(mean_dxhat_xhat, num_rows, last_dim));

    xla::XlaOp dgamma = xla::Reshape(
        SumAcrossRows2D(ctx, dy * x_hat, compute_type, num_rows, last_dim),
        DimSizes(gamma_shape));
    xla::XlaOp dbeta = xla::Reshape(
        SumAcrossRows2D(ctx, dy, compute_type, num_rows, last_dim),
        DimSizes(ctx->InputShape(3)));

    ctx->SetOutput(0, ConvertTo(xla::Reshape(dx_2d, DimSizes(x_shape)), out_type));
    ctx->SetOutput(1, ConvertTo(dgamma, out_type));
    ctx->SetOutput(2, ConvertTo(dbeta, out_type));
  }

 private:
  float epsilon_;
};

class MusaNormalizeXlaOp : public XlaOpKernel {
 public:
  explicit MusaNormalizeXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_std", &max_std_));
    if (max_std_ <= 0.0f) max_std_ = std::numeric_limits<float>::max();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    CompileNormalize(ctx, /*use_affine=*/false, epsilon_, max_std_,
                     /*layernorm_eps_inside_sqrt=*/false);
  }

 private:
  float epsilon_;
  float max_std_;
};

REGISTER_XLA_OP(Name("MusaLayerNorm").TypeConstraint("T", kFloatTypes),
                MusaLayerNormXlaOp);
REGISTER_XLA_OP(Name("MusaLayerNormGrad").TypeConstraint("T", kFloatTypes),
                MusaLayerNormGradXlaOp);
REGISTER_XLA_OP(Name("MusaNormalize").TypeConstraint("T", kFloatTypes),
                MusaNormalizeXlaOp);

}  // namespace
}  // namespace tensorflow
