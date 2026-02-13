#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSquaredDifferenceOp : public MusaOpKernel {
 public:
  explicit MusaSquaredDifferenceOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // --- 广播形状计算逻辑  ---
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

    // 判空处理
    if (in0.NumElements() == 0 || in1.NumElements() == 0 ||
        output_shape.num_elements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    // CreateMTensor 会自动处理 float/half/bfloat16 的类型映射
    mTensor t0 = CreateMTensor(in0, format_);
    mTensor t1 = CreateMTensor(in1, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    // --- MUSA 逻辑: (in0 - in1)^2 ---

    // 第一步: SUB
    ::musa::dnn::Binary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    auto status = binary_op.Run(handle, t_out, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Binary SUB execution failed."));

    // 第二步: SQUARE (原位计算)
    ::musa::dnn::Unary unary_op;
    unary_op.SetMode(::musa::dnn::Unary::Mode::SQUARE);
    status = unary_op.Run(handle, t_out, t_out);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Unary SQUARE execution failed."));
  }
};

// 注册算子
#define REGISTER_MUSA_SQUARED_DIFF(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SquaredDifference").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSquaredDifferenceOp<TYPE>);

REGISTER_MUSA_SQUARED_DIFF(float);
REGISTER_MUSA_SQUARED_DIFF(int32);
REGISTER_MUSA_SQUARED_DIFF(int64);
REGISTER_MUSA_SQUARED_DIFF(Eigen::half);
REGISTER_MUSA_SQUARED_DIFF(bfloat16);

}  // namespace musa
}  // namespace tensorflow
