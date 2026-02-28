#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

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

    const int dims0 = in0.dims();
    const int dims1 = in1.dims();
    const int out_dims = std::max(dims0, dims1);
    TensorShape output_shape;

    for (int i = 0; i < out_dims; ++i) {
      int d0 =
          (i < out_dims - dims0) ? 1 : in0.dim_size(i - (out_dims - dims0));
      int d1 =
          (i < out_dims - dims1) ? 1 : in1.dim_size(i - (out_dims - dims1));
      if (d0 == d1 || d0 == 1 || d1 == 1) {
        output_shape.AddDim(std::max(d0, d1));
      } else {
        ctx->CtxFailure(errors::InvalidArgument("Incompatible shapes"));
        return;
      }
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::SUB);

    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Sub execution failed."));
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
