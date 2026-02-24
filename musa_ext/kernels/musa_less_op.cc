#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLessOp : public MusaOpKernel {
 public:
  explicit MusaLessOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

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

      if (d0 == d1) {
        output_shape.AddDim(d0);
      } else if (d0 == 1) {
        output_shape.AddDim(d1);
      } else if (d1 == 1) {
        output_shape.AddDim(d0);
      } else {
        ctx->CtxFailure(errors::InvalidArgument(
            "Incompatible shapes: ", in0.shape().DebugString(), " and ",
            in1.shape().DebugString()));
        return;
      }
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (in0.NumElements() == 0 || in1.NumElements() == 0 ||
        output_shape.num_elements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0, format_);
    mTensor t1 = CreateMTensor(in1, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::LT);

    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Less execution failed."));
  }
};

#define REGISTER_MUSA_LESS(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Less").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      MusaLessOp<TYPE>);

REGISTER_MUSA_LESS(float);
REGISTER_MUSA_LESS(int32);
REGISTER_MUSA_LESS(int64);
REGISTER_MUSA_LESS(Eigen::half);
REGISTER_MUSA_LESS(bfloat16);
REGISTER_MUSA_LESS(double);
REGISTER_MUSA_LESS(uint8);

}  // namespace musa
}  // namespace tensorflow