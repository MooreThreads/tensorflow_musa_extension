#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <::musa::dnn::Binary::Mode mode>
class MusaComparisonOp : public MusaOpKernel {
 public:
  explicit MusaComparisonOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Comparison ops are element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument("Incompatible shapes for comparison op: ",
                                in0.shape().DebugString(), " vs ",
                                in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    ::musa::dnn::Binary op;
    auto status = op.SetMode(mode);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode failed"));

    status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Comparison Run failed"));
  }
};

using MusaEqualOp = MusaComparisonOp<::musa::dnn::Binary::Mode::EQ>;
using MusaNotEqualOp = MusaComparisonOp<::musa::dnn::Binary::Mode::NE>;
using MusaGreaterEqualOp = MusaComparisonOp<::musa::dnn::Binary::Mode::GE>;
using MusaLessEqualOp = MusaComparisonOp<::musa::dnn::Binary::Mode::LE>;

#define REGISTER_COMPPARISON_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Equal").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),        \
      MusaEqualOp);                                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("NotEqual").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),     \
      MusaNotEqualOp);                                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("GreaterEqual").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaGreaterEqualOp);                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LessEqual").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),    \
      MusaLessEqualOp);

REGISTER_COMPPARISON_KERNELS(float);        // FP32
REGISTER_COMPPARISON_KERNELS(double);       // FP64
REGISTER_COMPPARISON_KERNELS(int32);        // INT32
REGISTER_COMPPARISON_KERNELS(int64);        // INT64
REGISTER_COMPPARISON_KERNELS(Eigen::half);  // FP16
REGISTER_COMPPARISON_KERNELS(bfloat16);     // BF16

}  // namespace musa
}  // namespace tensorflow