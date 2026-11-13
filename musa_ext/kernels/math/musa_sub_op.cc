#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSubOp : public MusaOpKernel {
 public:
  explicit MusaSubOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Sub is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // Broadcasting 0 with 1 produces 0, not max(0, 1). Validate before
    // returning an empty output so incompatible empty inputs still fail.
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for Sub: ", in0.shape().DebugString(),
                    " and ", in1.shape().DebugString()));
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    if (out->NumElements() == 0) return;

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);

    // This overload exposes scalars to muDNN as [1] while preserving the
    // TensorFlow tensor's rank-0 shape.
    mTensor t0 = CreateMTensor(in0, mFormat::NCHW);
    mTensor t1 = CreateMTensor(in1, mFormat::NCHW);
    mTensor t_out = CreateMTensor(*out, mFormat::NCHW);

    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::SUB);

    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA Sub execution failed: node=", name(),
                    ", input shapes=", in0.shape().DebugString(), " and ",
                    in1.shape().DebugString(),
                    ", output shape=", out->shape().DebugString(),
                    ", dtype=", DataTypeString(in0.dtype()),
                    ", muDNN status=", static_cast<int>(status)));
  }
};

#define REGISTER_MUSA_SUB(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Sub").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"),   \
      MusaSubOp<TYPE>);                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SubV2").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaSubOp<TYPE>);

REGISTER_MUSA_SUB(float);
REGISTER_MUSA_SUB(int32);
REGISTER_MUSA_SUB(int64);
REGISTER_MUSA_SUB(Eigen::half);  // FP16
REGISTER_MUSA_SUB(bfloat16);     // BF16
REGISTER_MUSA_SUB(double);

}  // namespace musa
}  // namespace tensorflow
