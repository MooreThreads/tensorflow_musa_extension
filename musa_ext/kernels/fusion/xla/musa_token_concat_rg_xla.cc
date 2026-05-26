#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaTokenMixerXlaOp : public XlaOpKernel {
 public:
  explicit MusaTokenMixerXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_T", &num_t_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_H", &num_h_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("d_k", &d_k_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape x_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, x_shape.dims() == 3,
                errors::InvalidArgument("MusaTokenMixer requires rank-3 input"));
    const int64_t batch = x_shape.dim_size(0);
    OP_REQUIRES(ctx, x_shape.dim_size(1) == num_t_ &&
                         x_shape.dim_size(2) == num_h_ * d_k_,
                errors::InvalidArgument("MusaTokenMixer input shape does not "
                                        "match attributes"));
    xla::XlaOp x4 = xla::Reshape(ctx->Input(0), {batch, num_t_, num_h_, d_k_});
    xla::XlaOp y4 = xla::Transpose(x4, {0, 2, 1, 3});
    ctx->SetOutput(0, xla::Reshape(y4, {batch, num_h_, num_t_ * d_k_}));
  }

 private:
  int64_t num_t_;
  int64_t num_h_;
  int64_t d_k_;
};

class MusaConcatMatMulXlaOp : public XlaOpKernel {
 public:
  explicit MusaConcatMatMulXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_concat", &num_concat_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_input_idx", &concat_input_idx_));
    OP_REQUIRES(ctx, concat_input_idx_ == 0 || concat_input_idx_ == 1,
                errors::InvalidArgument("concat_input_idx must be 0 or 1"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> inputs;
    std::vector<TensorShape> input_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("inputs", &inputs, &input_shapes));
    OP_REQUIRES(ctx, static_cast<int>(inputs.size()) == num_concat_,
                errors::InvalidArgument("MusaConcatMatMul num_concat mismatch"));
    int64_t axis = 0;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(num_concat_, &axis));
    if (axis < 0) axis += input_shapes[0].dims();
    OP_REQUIRES(ctx, axis >= 0 && axis < input_shapes[0].dims(),
                errors::InvalidArgument("MusaConcatMatMul axis out of range"));

    xla::XlaOp concat = xla::ConcatInDim(ctx->builder(), inputs, axis);
    xla::XlaOp other = ctx->Input(num_concat_ + 1);
    xla::XlaOp lhs = concat_input_idx_ == 0 ? concat : other;
    xla::XlaOp rhs = concat_input_idx_ == 0 ? other : concat;
    ctx->SetOutput(0, BatchMatMul(lhs, transpose_a_, rhs, transpose_b_));
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  int num_concat_;
  int concat_input_idx_;
};

class MusaBiasAddReluMatMulXlaOp : public XlaOpKernel {
 public:
  explicit MusaBiasAddReluMatMulXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu_input_slot", &relu_input_slot_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES(ctx, relu_input_slot_ == 0 || relu_input_slot_ == 1,
                errors::InvalidArgument("relu_input_slot must be 0 or 1"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp bias = ctx->Input(1);
    BroadcastToShape(ctx, &bias, ctx->InputShape(0));
    xla::XlaOp bias_relu = xla::Max(input + bias, xla::ZerosLike(input));
    xla::XlaOp other = ctx->Input(2);
    xla::XlaOp lhs = relu_input_slot_ == 0 ? bias_relu : other;
    xla::XlaOp rhs = relu_input_slot_ == 0 ? other : bias_relu;
    ctx->SetOutput(0, BatchMatMul(lhs, transpose_a_, rhs, transpose_b_));
  }

 private:
  int relu_input_slot_;
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_XLA_OP(Name("MusaTokenMixer").TypeConstraint("T", kFloatTypes),
                MusaTokenMixerXlaOp);
REGISTER_XLA_OP(Name("MusaConcatMatMul")
                    .TypeConstraint("T", kFloatTypes)
                    .CompileTimeConstantInput("axis"),
                MusaConcatMatMulXlaOp);
REGISTER_XLA_OP(Name("MusaBiasAddReluMatMul").TypeConstraint("T", kFloatTypes),
                MusaBiasAddReluMatMulXlaOp);

}  // namespace
}  // namespace tensorflow
