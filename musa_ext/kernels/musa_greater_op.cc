#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h" // 必须引入广播工具类
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGreaterOp : public MusaOpKernel {
 public:
  explicit MusaGreaterOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);

    // 1. 处理广播逻辑，计算输出形状
    BCast bcast(BCast::FromShape(x.shape()), BCast::FromShape(y.shape()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for Greater: ",
                                        x.shape().DebugString(), " and ",
                                        y.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, BCast::ToShape(bcast.output_shape()), &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    // 2. 创建 MUSA Tensor

    mTensor mt_x = CreateMTensor(x);
    mTensor mt_y = CreateMTensor(y);
    mTensor mt_out = CreateMTensor(*output);

    // 3. 配置并执行 Binary 算子
    mBinary op;
    op.SetMode(::musa::dnn::Binary::Mode::GT); // 设置为 Greater Than 模式
    
    // muDNN 底层会根据 mt_x, mt_y 和 mt_out 的 shape 自动进行广播计算
    auto status = op.Run(handle, mt_out, mt_x, mt_y);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Greater execution failed. Status: ", (int)status));
  }
};

// 注册支持的类型
#define REGISTER_MUSA_GREATER(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Greater").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      MusaGreaterOp<TYPE>)

REGISTER_MUSA_GREATER(float);
REGISTER_MUSA_GREATER(Eigen::half);
REGISTER_MUSA_GREATER(int32);
REGISTER_MUSA_GREATER(int64);

#undef REGISTER_MUSA_GREATER

} // namespace musa
} // namespace tensorflow
