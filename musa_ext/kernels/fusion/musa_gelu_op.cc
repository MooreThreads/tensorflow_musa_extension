#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGeluOp : public MusaOpKernel {
 public:
  explicit MusaGeluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) {
      // VLOG(1) << "MusaGeluOp::Compute skipped empty tensor";
      return;
    }

    const int64 num_elements = input.NumElements();
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);
    mUnary op;
    const UNARY_MODE mode =
        approximate_ ? UNARY_MODE::GELU_TANH : UNARY_MODE::GELU;

    VLOG(1) << "MusaGeluOp::Compute launching muDNN GELU, elements="
            << num_elements << ", approximate=" << approximate_
            << ", mode=" << (approximate_ ? "GELU_TANH" : "GELU");

    MTOP_CHECK_OK(op.SetMode(mode), "Set GELU Mode", ctx);

    MTOP_CHECK_OK_RUN(op.Run(handle, mt_output, mt_input), "GELU Forward Run",
                      ctx);

    // VLOG(1) << "MusaGeluOp::Compute finished, elements=" << num_elements
    //         << ", approximate=" << approximate_;
  }

 private:
  bool approximate_;
};

#define REGISTER_MUSA_GELU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MusaGelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaGeluOp<TYPE>);

REGISTER_MUSA_GELU(float);
REGISTER_MUSA_GELU(double);
REGISTER_MUSA_GELU(Eigen::half);
REGISTER_MUSA_GELU(bfloat16);

#undef REGISTER_MUSA_GELU

}  // namespace musa

REGISTER_OP("MusaGelu")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("approximate: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
