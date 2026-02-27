#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaFloorModOp : public MusaOpKernel {
 public:
  explicit MusaFloorModOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " and ",
                    in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

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

    ::musa::dnn::Binary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::FLOORMOD);

    auto status = binary_op.Run(handle, t_out, t0, t1);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Native FLOORMOD execution failed. Status code: ",
                         static_cast<int>(status)));
  }
};

#define REGISTER_MUSA_FLOORMOD(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FloorMod").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFloorModOp<TYPE>);

REGISTER_MUSA_FLOORMOD(float);
REGISTER_MUSA_FLOORMOD(double);
REGISTER_MUSA_FLOORMOD(Eigen::half);
REGISTER_MUSA_FLOORMOD(bfloat16);
REGISTER_MUSA_FLOORMOD(int32);
REGISTER_MUSA_FLOORMOD(int64);

#undef REGISTER_MUSA_FLOORMOD

}  // namespace musa
}  // namespace tensorflow
