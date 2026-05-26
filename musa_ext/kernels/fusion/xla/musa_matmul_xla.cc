#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaMatMulBiasAddXlaOp : public XlaOpKernel {
 public:
  explicit MusaMatMulBiasAddXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape a_shape = ctx->InputShape(0);
    const TensorShape b_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, a_shape.dims() == 2 && b_shape.dims() == 2,
                errors::InvalidArgument("MusaMatMulBiasAdd requires rank-2 "
                                        "matrix inputs"));
    const int64_t m = transpose_a_ ? a_shape.dim_size(1) : a_shape.dim_size(0);
    const int64_t k_a = transpose_a_ ? a_shape.dim_size(0) : a_shape.dim_size(1);
    const int64_t k_b = transpose_b_ ? b_shape.dim_size(1) : b_shape.dim_size(0);
    const int64_t n = transpose_b_ ? b_shape.dim_size(0) : b_shape.dim_size(1);
    OP_REQUIRES(ctx, k_a == k_b,
                errors::InvalidArgument("MusaMatMulBiasAdd matrix size "
                                        "incompatible"));

    TensorShape out_shape({m, n});
    xla::XlaOp output =
        BatchMatMul(ctx->Input(0), transpose_a_, ctx->Input(1), transpose_b_);
    xla::XlaOp bias = ctx->Input(2);
    BroadcastToShape(ctx, &bias, out_shape);
    ctx->SetOutput(0, output + bias);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

class MusaLinearActivationXlaOp : public XlaOpKernel {
 public:
  explicit MusaLinearActivationXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation", &activation_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES(ctx, activation_ == "relu",
                errors::InvalidArgument("Unsupported MusaLinearActivation "
                                        "activation: ",
                                        activation_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp output =
        BatchMatMul(ctx->Input(0), transpose_a_, ctx->Input(1), transpose_b_);
    TensorShape out_shape;
    ShapeOfXlaOp(ctx, output, &out_shape);
    xla::XlaOp bias = ctx->Input(2);
    BroadcastToShape(ctx, &bias, out_shape);
    ctx->SetOutput(0, xla::Max(output + bias, xla::ZerosLike(output)));
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  std::string activation_;
  float alpha_;
};

class MusaReshapeMatMulXlaOp : public XlaOpKernel {
 public:
  explicit MusaReshapeMatMulXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape w_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, x_shape.dims() >= 2 && w_shape.dims() == 2,
                errors::InvalidArgument("MusaReshapeMatMul requires x rank >= "
                                        "2 and w rank == 2"));
    const int64_t k = x_shape.dim_size(x_shape.dims() - 1);
    const int64_t w_k = transpose_b_ ? w_shape.dim_size(1) : w_shape.dim_size(0);
    const int64_t n = transpose_b_ ? w_shape.dim_size(0) : w_shape.dim_size(1);
    OP_REQUIRES(ctx, k == w_k,
                errors::InvalidArgument("MusaReshapeMatMul matrix size "
                                        "incompatible"));

    const int64_t m = x_shape.num_elements() / k;
    xla::XlaOp x_2d = xla::Reshape(ctx->Input(0), {m, k});
    xla::XlaOp y_2d =
        BatchMatMul(x_2d, /*transpose_a=*/false, ctx->Input(1), transpose_b_);
    std::vector<int64_t> out_dims = DimSizes(x_shape);
    out_dims.back() = n;
    ctx->SetOutput(0, xla::Reshape(y_2d, out_dims));
  }

 private:
  bool transpose_b_;
};

REGISTER_XLA_OP(Name("MusaMatMulBiasAdd").TypeConstraint("T", kFloatTypes),
                MusaMatMulBiasAddXlaOp);
REGISTER_XLA_OP(Name("MusaLinearActivation").TypeConstraint("T", kFloatTypes),
                MusaLinearActivationXlaOp);
REGISTER_XLA_OP(Name("MusaReshapeMatMul").TypeConstraint("T", kFloatTypes),
                MusaReshapeMatMulXlaOp);

}  // namespace
}  // namespace tensorflow
