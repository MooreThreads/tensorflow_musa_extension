#include "musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaShiftedAffineMapXlaOp : public XlaOpKernel {
 public:
  explicit MusaShiftedAffineMapXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    BCast bcast_lm(BCast::FromShape(ctx->InputShape(0)),
                   BCast::FromShape(ctx->InputShape(1)));
    OP_REQUIRES(ctx, bcast_lm.IsValid(),
                errors::InvalidArgument("MusaShiftedAffineMap lhs/mask shapes "
                                        "are not broadcast-compatible"));
    BCast bcast_all(bcast_lm.output_shape(),
                    BCast::FromShape(ctx->InputShape(2)));
    OP_REQUIRES(ctx, bcast_all.IsValid(),
                errors::InvalidArgument("MusaShiftedAffineMap shapes are not "
                                        "broadcast-compatible"));
    TensorShape out_shape = BCast::ToShape(bcast_all.output_shape());
    xla::XlaOp data_left = ctx->Input(0);
    xla::XlaOp mask = ctx->Input(1);
    xla::XlaOp right = ctx->Input(2);
    BroadcastToShape(ctx, &data_left, out_shape);
    BroadcastToShape(ctx, &mask, out_shape);
    BroadcastToShape(ctx, &right, out_shape);
    ctx->SetOutput(0, mask * data_left + right);
  }
};

class MusaPlnCascadeXlaOp : public XlaOpKernel {
 public:
  explicit MusaPlnCascadeXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_table", &use_table_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_index", &table_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_on_true", &select_on_true_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape out_shape;
    bool mask_left_aligned = false;
    OP_REQUIRES(ctx,
                BroadcastShapeOrLeftAlignedMask(
                    ctx->InputShape(0), ctx->InputShape(1), &out_shape,
                    &mask_left_aligned),
                errors::InvalidArgument(
                    "MusaPlnCascade mask is not broadcast-compatible"));

    xla::XlaOp norm = ctx->Input(0);
    xla::XlaOp mask = ctx->Input(1);
    BroadcastToShape(ctx, &norm, out_shape);
    if (mask_left_aligned) {
      BroadcastLeftAligned1DToShape(ctx, &mask, ctx->InputShape(1), out_shape);
    } else {
      BroadcastToShape(ctx, &mask, out_shape);
    }

    xla::XlaOp add;
    xla::XlaOp bias;
    if (use_table_) {
      TableRow(ctx, ctx->Input(2), ctx->InputShape(2), table_index_, &add);
      TableRow(ctx, ctx->Input(3), ctx->InputShape(3), table_index_, &bias);
      OP_REQUIRES(ctx, out_shape.dims() >= 1 &&
                           out_shape.dim_size(out_shape.dims() - 1) ==
                               ctx->InputShape(2).dim_size(1),
                  errors::InvalidArgument(
                      "MusaPlnCascade table width must match output last dim"));
      BroadcastToShape(ctx, &add, out_shape);
      BroadcastToShape(ctx, &bias, out_shape);
    } else {
      BCast bcast_add(BCast::FromShape(out_shape),
                      BCast::FromShape(ctx->InputShape(2)));
      OP_REQUIRES(ctx, bcast_add.IsValid(),
                  errors::InvalidArgument(
                      "MusaPlnCascade add_input is not broadcast-compatible"));
      out_shape = BCast::ToShape(bcast_add.output_shape());
      BCast bcast_bias(BCast::FromShape(out_shape),
                       BCast::FromShape(ctx->InputShape(3)));
      OP_REQUIRES(ctx, bcast_bias.IsValid(),
                  errors::InvalidArgument(
                      "MusaPlnCascade bias_input is not broadcast-compatible"));
      out_shape = BCast::ToShape(bcast_bias.output_shape());

      BroadcastToShape(ctx, &norm, out_shape);
      BroadcastMaskToShape(ctx, &mask, ctx->InputShape(1), out_shape);
      add = ctx->Input(2);
      bias = ctx->Input(3);
      BroadcastToShape(ctx, &add, out_shape);
      BroadcastToShape(ctx, &bias, out_shape);
    }

    xla::XlaOp candidate = norm * add + bias;
    xla::XlaOp take_candidate = select_on_true_ ? mask : xla::Not(mask);
    ctx->SetOutput(0, xla::Select(take_candidate, candidate, norm));
  }

 private:
  bool use_table_;
  int table_index_;
  bool select_on_true_;
};

class MusaPlnCascadeBlockXlaOp : public XlaOpKernel {
 public:
  explicit MusaPlnCascadeBlockXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &n_steps_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_indices", &table_indices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_on_true", &select_on_true_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> gates;
    std::vector<TensorShape> gate_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("gates", &gates, &gate_shapes));
    OP_REQUIRES(ctx, static_cast<int>(gates.size()) == n_steps_,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock gates size does not match N"));
    OP_REQUIRES(ctx, table_indices_.size() == gates.size() &&
                         select_on_true_.size() == gates.size(),
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock attr list sizes must match N"));
    OP_REQUIRES(ctx, ctx->InputShape(1).dims() == 2 &&
                         ctx->InputShape(2).dims() == 2 &&
                         ctx->InputShape(1) == ctx->InputShape(2),
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock add/bias tables must be rank-2 and "
                    "same shape"));

    TensorShape out_shape = ctx->InputShape(0);
    std::vector<char> gate_left_aligned(gates.size(), 0);
    for (size_t i = 0; i < gates.size(); ++i) {
      TensorShape next_shape;
      bool left_aligned = false;
      OP_REQUIRES(ctx,
                  BroadcastShapeOrLeftAlignedMask(out_shape, gate_shapes[i],
                                                  &next_shape, &left_aligned),
                  errors::InvalidArgument(
                      "MusaPlnCascadeBlock gate is not broadcast-compatible"));
      out_shape = next_shape;
      gate_left_aligned[i] = left_aligned ? 1 : 0;
    }
    OP_REQUIRES(ctx, out_shape.dims() >= 1 &&
                         out_shape.dim_size(out_shape.dims() - 1) ==
                             ctx->InputShape(1).dim_size(1),
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock table width must match output last "
                    "dim"));

    xla::XlaOp value = ctx->Input(0);
    BroadcastToShape(ctx, &value, out_shape);
    for (size_t i = 0; i < gates.size(); ++i) {
      xla::XlaOp gate = gates[i];
      if (gate_left_aligned[i] != 0) {
        BroadcastLeftAligned1DToShape(ctx, &gate, gate_shapes[i], out_shape);
      } else {
        BroadcastToShape(ctx, &gate, out_shape);
      }

      xla::XlaOp add;
      xla::XlaOp bias;
      TableRow(ctx, ctx->Input(1), ctx->InputShape(1), table_indices_[i], &add);
      TableRow(ctx, ctx->Input(2), ctx->InputShape(2), table_indices_[i],
               &bias);
      BroadcastToShape(ctx, &add, out_shape);
      BroadcastToShape(ctx, &bias, out_shape);
      xla::XlaOp candidate = value * add + bias;
      xla::XlaOp take_candidate = select_on_true_[i] ? gate : xla::Not(gate);
      value = xla::Select(take_candidate, candidate, value);
    }

    ctx->SetOutput(0, value);
  }

 private:
  int n_steps_;
  std::vector<int> table_indices_;
  std::vector<bool> select_on_true_;
};

REGISTER_XLA_OP(Name("MusaShiftedAffineMap").TypeConstraint("T", kFloatTypes),
                MusaShiftedAffineMapXlaOp);
REGISTER_XLA_OP(Name("MusaPlnCascade").TypeConstraint("T", DT_FLOAT),
                MusaPlnCascadeXlaOp);
REGISTER_XLA_OP(Name("MusaPlnCascadeBlock").TypeConstraint("T", DT_FLOAT),
                MusaPlnCascadeBlockXlaOp);

}  // namespace
}  // namespace tensorflow
