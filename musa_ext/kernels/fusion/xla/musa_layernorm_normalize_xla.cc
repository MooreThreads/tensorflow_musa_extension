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
    OP_REQUIRES(ctx, gamma_shape.num_elements() == last_dim &&
                         ctx->InputShape(3).num_elements() == last_dim,
                errors::InvalidArgument(
                    "MusaLayerNormGrad gamma/beta size must match last dim"));

    const DataType out_type = ctx->input_type(0);
    const DataType compute_type = out_type == DT_DOUBLE ? DT_DOUBLE : DT_FLOAT;
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp dy = ConvertTo(ctx->Input(0), compute_type);
    xla::XlaOp x = ConvertTo(ctx->Input(1), compute_type);
    xla::XlaOp gamma = ConvertTo(ctx->Input(2), compute_type);
    std::vector<int64_t> x_dims = DimSizes(x_shape);
    std::vector<int64_t> leading_dims = IotaDims(rank - 1);

    xla::XlaOp sum =
        xla::Reduce(x, XlaHelpers::Zero(b, compute_type),
                    *ctx->GetOrCreateAdd(compute_type), {rank - 1});
    xla::XlaOp mean = sum / XlaHelpers::FloatLiteral(b, compute_type, last_dim);
    xla::XlaOp mean_b = xla::BroadcastInDim(mean, x_dims, leading_dims);
    xla::XlaOp centered = x - mean_b;
    xla::XlaOp var_sum =
        xla::Reduce(centered * centered, XlaHelpers::Zero(b, compute_type),
                    *ctx->GetOrCreateAdd(compute_type), {rank - 1});
    xla::XlaOp variance =
        var_sum / XlaHelpers::FloatLiteral(b, compute_type, last_dim);
    xla::XlaOp inv_std = xla::Rsqrt(
        variance + XlaHelpers::FloatLiteral(b, compute_type, epsilon_));
    xla::XlaOp inv_std_b = xla::BroadcastInDim(inv_std, x_dims, leading_dims);
    xla::XlaOp x_hat = centered * inv_std_b;

    xla::XlaOp gamma_b = gamma;
    BroadcastToShape(ctx, &gamma_b, x_shape);
    xla::XlaOp dxhat = dy * gamma_b;
    xla::XlaOp mean_dxhat =
        xla::Reduce(dxhat, XlaHelpers::Zero(b, compute_type),
                    *ctx->GetOrCreateAdd(compute_type), {rank - 1}) /
        XlaHelpers::FloatLiteral(b, compute_type, last_dim);
    xla::XlaOp mean_dxhat_xhat =
        xla::Reduce(dxhat * x_hat, XlaHelpers::Zero(b, compute_type),
                    *ctx->GetOrCreateAdd(compute_type), {rank - 1}) /
        XlaHelpers::FloatLiteral(b, compute_type, last_dim);
    xla::XlaOp mean_dxhat_b =
        xla::BroadcastInDim(mean_dxhat, x_dims, leading_dims);
    xla::XlaOp mean_dxhat_xhat_b =
        xla::BroadcastInDim(mean_dxhat_xhat, x_dims, leading_dims);
    xla::XlaOp dx =
        inv_std_b * (dxhat - mean_dxhat_b - x_hat * mean_dxhat_xhat_b);

    std::vector<int64_t> reduce_axes = IotaDims(rank - 1);
    xla::XlaOp dgamma = dy * x_hat;
    xla::XlaOp dbeta = dy;
    if (!reduce_axes.empty()) {
      dgamma = xla::Reduce(dgamma, XlaHelpers::Zero(b, compute_type),
                           *ctx->GetOrCreateAdd(compute_type), reduce_axes);
      dbeta = xla::Reduce(dbeta, XlaHelpers::Zero(b, compute_type),
                          *ctx->GetOrCreateAdd(compute_type), reduce_axes);
    }
    dgamma = xla::Reshape(dgamma, DimSizes(gamma_shape));
    dbeta = xla::Reshape(dbeta, DimSizes(ctx->InputShape(3)));

    ctx->SetOutput(0, ConvertTo(dx, out_type));
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
