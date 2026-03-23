#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

timing::KernelTimingLayout LogicalBinaryTimingLayout() {
  return MUSA_KERNEL_TIMING_LAYOUT(
      MUSA_KERNEL_TIMING_STAGE("BCast", "BCast", false),
      MUSA_KERNEL_TIMING_STAGE("Alloc", "Alloc", false),
      MUSA_KERNEL_TIMING_STAGE("Tensor Wrap", "Tensor Wrap", false),
      MUSA_KERNEL_TIMING_STAGE("SetMode", "SetMode", false),
      MUSA_KERNEL_TIMING_STAGE("Kernel", "Kernel", false));
}

}  // namespace

template <::musa::dnn::Binary::Mode mode>
class MusaLogicalBinaryOp : public MusaOpKernel {
 public:
  explicit MusaLogicalBinaryOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_LAYOUT(ctx, LogicalBinaryTimingLayout());
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    MUSA_KERNEL_TRACE_START("BCast");
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for logical op: ",
                                        in0.shape().DebugString(), " vs ",
                                        in1.shape().DebugString()));
    const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    MUSA_KERNEL_TRACE_END("BCast");

    MUSA_KERNEL_TRACE_START("Alloc");
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    MUSA_KERNEL_TRACE_END("Alloc");

    if (out->NumElements() == 0) return;

    MUSA_KERNEL_TRACE_START("Tensor Wrap");
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);
    MUSA_KERNEL_TRACE_END("Tensor Wrap");

    ::musa::dnn::Binary op;

    MUSA_KERNEL_TRACE_START("SetMode");
    auto status = op.SetMode(mode);
    MUSA_KERNEL_TRACE_END("SetMode");
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode failed for logical op"));

    MUSA_KERNEL_TRACE_START("Kernel");
    status = op.Run(handle, t_out, t0, t1);
    MUSA_KERNEL_TRACE_END("Kernel");

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
