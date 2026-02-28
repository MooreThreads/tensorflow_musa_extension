#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMinimumOp : public MusaOpKernel {
 public:
  explicit MusaMinimumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Minimum is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &out));

    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;
    binary_op.SetMode(BINARY_MODE::MIN);

    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*out, format_);

    mStatus status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);

    OP_REQUIRES(
        ctx, status == mStatus::SUCCESS,
        errors::Internal("muDNN Minimum execution failed. Status code: ",
                         (int)status));
  }
};

}  // namespace musa
}  // namespace tensorflow

using namespace tensorflow;

#define REGISTER_MUSA_MIN(type)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Minimum").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      ::tensorflow::musa::MusaMinimumOp<type>);

REGISTER_MUSA_MIN(float);
REGISTER_MUSA_MIN(double);
REGISTER_MUSA_MIN(int32);
REGISTER_MUSA_MIN(int64);
REGISTER_MUSA_MIN(Eigen::half);  // FP16
REGISTER_MUSA_MIN(bfloat16);     // BF16

#undef REGISTER_MUSA_MIN
