#include <mudnn.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaMaximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, half, bfloat16, double, int32, int64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

template <typename T>
class MusaMaximumOp : public MusaOpKernel {
 public:
  explicit MusaMaximumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_x = ctx->input(0);
    const Tensor& input_y = ctx->input(1);

    BCast bcast(BCast::Vec(input_x.shape().dim_sizes()),
                BCast::Vec(input_y.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_x.shape().DebugString(),
                    " vs ", input_y.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;

    binary_op.SetMode(BINARY_MODE::MAX);

    mTensor mt_x = CreateMTensor(input_x, format_);
    mTensor mt_y = CreateMTensor(input_y, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    auto status = binary_op.Run(handle, mt_out, mt_x, mt_y);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Maximum execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MAXIMUM(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Maximum").Device("MUSA").TypeConstraint<TYPE>("T"),     \
      MusaMaximumOp<TYPE>);                                         \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("MusaMaximum").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMaximumOp<TYPE>);

REGISTER_MAXIMUM(float);
REGISTER_MAXIMUM(Eigen::half);
REGISTER_MAXIMUM(bfloat16);
REGISTER_MAXIMUM(int32);
REGISTER_MAXIMUM(int64);

#undef REGISTER_MAXIMUM

}  // namespace musa
}  // namespace tensorflow
