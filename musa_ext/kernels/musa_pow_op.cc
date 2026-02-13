#include <mudnn.h>

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaPowOp : public MusaOpKernel {
 public:
  explicit MusaPowOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape()),
                true);

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
                    in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (output_shape.num_elements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t0 = CreateMTensor(in0, format_);
    mTensor t1 = CreateMTensor(in1, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    const auto& s0 = bcast.x_reshape();
    const auto& s1 = bcast.y_reshape();
    const auto& s_out = bcast.result_shape();

    std::vector<int64_t> v0(s0.begin(), s0.end());
    std::vector<int64_t> v1(s1.begin(), s1.end());
    std::vector<int64_t> v_out(s_out.begin(), s_out.end());

    OP_REQUIRES(ctx,
                t0.SetNdInfo(static_cast<int>(v0.size()), v0.data()) ==
                    mStatus::SUCCESS,
                errors::Internal("SetNdInfo t0 failed"));
    OP_REQUIRES(ctx,
                t1.SetNdInfo(static_cast<int>(v1.size()), v1.data()) ==
                    mStatus::SUCCESS,
                errors::Internal("SetNdInfo t1 failed"));
    OP_REQUIRES(ctx,
                t_out.SetNdInfo(static_cast<int>(v_out.size()), v_out.data()) ==
                    mStatus::SUCCESS,
                errors::Internal("SetNdInfo t_out failed"));

    // 5. 执行 POW
    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::POW);

    auto status = op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Pow execution failed. Status: ",
                                 static_cast<int>(status)));
  }
};

#define REGISTER_MUSA_POW(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Pow").Device("MUSA").TypeConstraint<TYPE>("T"), MusaPowOp<TYPE>);

REGISTER_MUSA_POW(float);
REGISTER_MUSA_POW(int32);
REGISTER_MUSA_POW(int64);
REGISTER_MUSA_POW(Eigen::half);
REGISTER_MUSA_POW(bfloat16);
REGISTER_MUSA_POW(double);

}  // namespace musa
}  // namespace tensorflow
