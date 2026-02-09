/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include <mudnn.h>
#include <mudnn_tensor.h>
#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T>
class MusaZerosLikeOp : public MusaOpKernel {
 public:
  explicit MusaZerosLikeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    // 1. 创建形状一致的输出 Tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    // 如果元素数量为 0，直接返回
    if (output->NumElements() == 0) return;

    // 2. 获取 MUSA 句柄和 Tensor 描述符
    auto& h = GetHandleByCtx(ctx);
    auto out_mt = CreateMTensor(*output);

    // 3. 调用下沉后的 Fill 算子
   
    ::musa::dnn::Fill op;
    
    // 设置填充值为 0
    // 注意：Fill 算子的 SetValue 通常接收 double，内部会根据 Tensor 类型自动转换
    MTOP_CHECK_OK(op.SetValue(0.0), "Fill SetValue to 0", ctx);

    // 4. 执行填充
    MTOP_CHECK_OK_RUN(op.Run(h, out_mt), "Fill Run for ZerosLike", ctx);
  }
};

// =====================================================================
// 5. 注册算子 (支持 6 种常用类型)
// =====================================================================
#define REGISTER_MUSA_ZEROS_LIKE(type)                        \
  REGISTER_KERNEL_BUILDER(Name("ZerosLike")                   \
                              .Device(DEVICE_MTGPU)           \
                              .TypeConstraint<type>("T"),     \
                          MusaZerosLikeOp<type>);

REGISTER_MUSA_ZEROS_LIKE(float);
REGISTER_MUSA_ZEROS_LIKE(Eigen::half);
REGISTER_MUSA_ZEROS_LIKE(double);
REGISTER_MUSA_ZEROS_LIKE(int32);
REGISTER_MUSA_ZEROS_LIKE(int64);
REGISTER_MUSA_ZEROS_LIKE(bool);

#undef REGISTER_MUSA_ZEROS_LIKE

}  // namespace
}  // namespace musa
}  // namespace tensorflow
