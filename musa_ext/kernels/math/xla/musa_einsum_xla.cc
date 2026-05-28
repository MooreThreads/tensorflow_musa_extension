#include "fusion/xla/musa_fusion_xla_common.h"

namespace tensorflow {
namespace {

class MusaOneTrans3DEinsumXlaOp : public XlaOpKernel {
 public:
  explicit MusaOneTrans3DEinsumXlaOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equation", &equation_));
    OP_REQUIRES(ctx,
                equation_ == "btd,tde->bte" || equation_ == "bte,tde->btd" ||
                    equation_ == "bte,btd->tde",
                errors::InvalidArgument("Unsupported MusaOneTrans3DEinsum "
                                        "equation: ",
                                        equation_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->InputShape(0).dims() == 3 &&
                         ctx->InputShape(1).dims() == 3,
                errors::InvalidArgument(
                    "MusaOneTrans3DEinsum expects rank-3 inputs"));
    xla::PrecisionConfig::Precision precision =
        tsl::tensor_float_32_execution_enabled()
            ? xla::PrecisionConfig::DEFAULT
            : xla::PrecisionConfig::HIGHEST;
    ctx->SetOutput(0, xla::Einsum(ctx->Input(0), ctx->Input(1), equation_,
                                  precision,
                                  /*preferred_element_type=*/std::nullopt));
  }

 private:
  std::string equation_;
};

REGISTER_XLA_OP(Name("MusaOneTrans3DEinsum").TypeConstraint("T", kFloatTypes),
                MusaOneTrans3DEinsumXlaOp);

}  // namespace
}  // namespace tensorflow
