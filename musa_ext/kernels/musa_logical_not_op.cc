#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaLogicalNotOp : public MusaOpKernel {
 public:
  explicit MusaLogicalNotOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dtype() == DT_BOOL,
                errors::InvalidArgument("LogicalNot expects bool input, got ",
                                        DataTypeString(input.dtype())));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);

    // NOT(x) = XOR(x, True)
    Tensor true_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_BOOL, TensorShape({}), &true_tensor));
    true_tensor.scalar<bool>()() = true;
    auto true_mt = CreateMTensor(true_tensor, format_);

    ::musa::dnn::Binary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::LOGICAL_XOR),
                  "Set LOGICAL_XOR", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt, true_mt),
                      "LogicalNot via XOR Run", ctx);
  }
};

REGISTER_KERNEL_BUILDER(Name("LogicalNot").Device("MUSA"), MusaLogicalNotOp);

}  // namespace musa
}  // namespace tensorflow
