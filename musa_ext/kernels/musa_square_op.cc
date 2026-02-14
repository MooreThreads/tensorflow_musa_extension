#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSquareOp : public MusaOpKernel {
 public:
  explicit MusaSquareOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto mt_in = CreateMTensor(input, format_);
    auto mt_out = CreateMTensor(*output, format_);

    mBinary op;
    op.SetMode(::musa::dnn::Binary::Mode::MUL);

    auto status = op.Run(handle, mt_out, mt_in, mt_in);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Square execution failed. Status: ",
                                 static_cast<int>(status)));
  }
};

#define REGISTER_SQUARE(type)                                  \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Square").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSquareOp<type>);

REGISTER_SQUARE(float);
REGISTER_SQUARE(Eigen::half);
REGISTER_SQUARE(bfloat16);
REGISTER_SQUARE(int32);
REGISTER_SQUARE(int64);

#undef REGISTER_SQUARE

}  // namespace musa
}  // namespace tensorflow
