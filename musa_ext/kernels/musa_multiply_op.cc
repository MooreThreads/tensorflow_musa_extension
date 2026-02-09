#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h" // [重要] 引入广播工具
#include "utils_op.h" // 引用你提供的工具头文件

namespace tensorflow {
namespace musa {

// 模板类，支持 T 类型
template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  void Compute(OpKernelContext* ctx) override {
    //fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // 1. 处理广播逻辑 (Broadcasting)
    // Sigmoid 不需要这个，但 Multiply 需要处理如 (1, 5) * (5, 5) 的情况
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    // 检查形状是否兼容
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for Mul: ",
                                        in0.shape().DebugString(), " and ",
                                        in1.shape().DebugString()));

    // 计算输出形状
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    // 2. 分配输出内存
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // 如果元素个数为0，直接返回
    if (output->NumElements() == 0) return;

    // 3. 获取句柄
    auto& handle = GetHandleByCtx(ctx);

    // 4. 设置 Op 模式
  
    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL); // 设置模式为乘法

    // 5. 创建 mTensor
    // 直接调用 utils_op.h 中的 CreateMTensor，它内部已经处理了 dtype 到 mType 的转换
    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    // 6. 执行计算
    auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Multiply execution failed. Status: ", (int)status));
  }
};

// 7. 注册所有 6 种类型
#define REGISTER_MUSA_MULTIPLY(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("Mul")                                   \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaMultiplyOp<TYPE>);

REGISTER_MUSA_MULTIPLY(float);
//REGISTER_MUSA_MULTIPLY(double);
REGISTER_MUSA_MULTIPLY(Eigen::half);
REGISTER_MUSA_MULTIPLY(bfloat16);
REGISTER_MUSA_MULTIPLY(int32);
REGISTER_MUSA_MULTIPLY(int64);

#undef REGISTER_MUSA_MULTIPLY

} // namespace musa
} // namespace tensorflow

