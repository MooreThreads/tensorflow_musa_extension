#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGreaterOp : public MusaOpKernel {
 public:
  explicit MusaGreaterOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);

    BCast bcast(BCast::FromShape(x.shape()), BCast::FromShape(y.shape()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for Greater: ",
                                        x.shape().DebugString(), " and ",
                                        y.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_x = CreateMTensor(x);
    mTensor mt_y = CreateMTensor(y);
    mTensor mt_out = CreateMTensor(*output);

    mBinary op;
    op.SetMode(::musa::dnn::Binary::Mode::GT);

    auto status = op.Run(handle, mt_out, mt_x, mt_y);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Greater execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_GREATER(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Greater").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaGreaterOp<TYPE>)

REGISTER_MUSA_GREATER(float);
REGISTER_MUSA_GREATER(Eigen::half);
REGISTER_MUSA_GREATER(int32);
REGISTER_MUSA_GREATER(int64);

#undef REGISTER_MUSA_GREATER

}  // namespace musa
}  // namespace tensorflow
