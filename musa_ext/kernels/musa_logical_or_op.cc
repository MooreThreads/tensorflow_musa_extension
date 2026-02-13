#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <::musa::dnn::Binary::Mode mode>
class MusaLogicalBinaryOp : public MusaOpKernel {
 public:
  explicit MusaLogicalBinaryOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for logical op: ",
                                        in0.shape().DebugString(), " vs ",
                                        in1.shape().DebugString()));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &out));

    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    ::musa::dnn::Binary op;

    auto status = op.SetMode(mode);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode failed for logical op"));

    status = op.Run(handle, t_out, t0, t1);

    if (status != mStatus::SUCCESS) {
      LOG(ERROR) << "muDNN Logical binary op Run failed, status: "
                 << (int)status;
    }

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal(
                    "muDNN Logical Run failed. "
                    "Check if muDNN supports BOOL kernels for this mode."));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("LogicalOr").Device(DEVICE_MTGPU),
    MusaLogicalBinaryOp<::musa::dnn::Binary::Mode::LOGICAL_OR>);

REGISTER_KERNEL_BUILDER(
    Name("LogicalAnd").Device(DEVICE_MTGPU),
    MusaLogicalBinaryOp<::musa::dnn::Binary::Mode::LOGICAL_AND>);

}  // namespace musa
}  // namespace tensorflow
