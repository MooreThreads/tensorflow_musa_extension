#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_memset.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool IsScalarTensor(const Tensor& t) {
  return TensorShapeUtils::IsScalar(t.shape());
}

inline Status ReadScalarBoolFromDevice(const Tensor& scalar_tensor,
                                       bool* value) {
  auto status =
      MusaMemcpyD2H(value, scalar_tensor.flat<bool>().data(), sizeof(bool));
  if (status != mStatus::SUCCESS) {
    return errors::Internal("MUSA D2H copy failed for logical scalar input.");
  }
  return Status::OK();
}

template <::musa::dnn::Binary::Mode mode>
Status MaybeHandleScalarLogicalShortcut(OpKernelContext* ctx, const Tensor& in0,
                                        const Tensor& in1, bool* handled) {
  *handled = false;

  const Tensor* scalar_input = nullptr;
  const Tensor* other_input = nullptr;
  if (IsScalarTensor(in0)) {
    scalar_input = &in0;
    other_input = &in1;
  } else if (IsScalarTensor(in1)) {
    scalar_input = &in1;
    other_input = &in0;
  } else {
    return Status::OK();
  }

  bool scalar_value = false;
  TF_RETURN_IF_ERROR(ReadScalarBoolFromDevice(*scalar_input, &scalar_value));

  // For bool logical ops, scalar inputs collapse to either the other operand
  // or a fully constant output.
  const bool passthrough_other =
      mode == ::musa::dnn::Binary::Mode::LOGICAL_OR ? !scalar_value
                                                    : scalar_value;
  if (passthrough_other) {
    ctx->set_output(0, *other_input);
    *handled = true;
    return Status::OK();
  }

  Tensor* out = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_output(0, other_input->shape(), &out));
  if (out->NumElements() == 0) {
    *handled = true;
    return Status::OK();
  }

  const bool fill_value = mode == ::musa::dnn::Binary::Mode::LOGICAL_OR;
  const uint8_t pattern = fill_value ? 1 : 0;
  auto status = MemsetAsync(out->data(), pattern, out->TotalBytes(),
                            GetMusaStreamByCtx(ctx));
  if (status != mStatus::SUCCESS) {
    return errors::Internal("MUSA MemsetAsync failed for logical scalar "
                            "constant output.");
  }
  *handled = true;
  return Status::OK();
}

}  // namespace

template <::musa::dnn::Binary::Mode mode>
class MusaLogicalBinaryOp : public MusaOpKernel {
 public:
  explicit MusaLogicalBinaryOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    bool handled = false;
    OP_REQUIRES_OK(ctx,
                   MaybeHandleScalarLogicalShortcut<mode>(ctx, in0, in1,
                                                          &handled));
    if (handled) return;

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
