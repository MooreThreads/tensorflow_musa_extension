/* Copyright @2020-2026 Moore Threads. All rights reserved. */
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"
#include <mudnn.h>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMinimumOp : public MusaOpKernel { // 继承 MusaOpKernel 以获取 format_
 public:
  explicit MusaMinimumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

   
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &out));

    if (out->NumElements() == 0) return;

    // --- 步骤 1: 获取 muDNN 句柄 ---
    auto& handle = GetHandleByCtx(ctx);

    // --- 步骤 2: 准备 muDNN 算子并设置为 MIN 模式 ---
    mBinary binary_op;
    // 使用 BINARY_MODE::MIN，该定义通常在 utils_op.h 或 mudnn.h 中
    binary_op.SetMode(BINARY_MODE::MIN);

    // --- 步骤 3: 创建 muDNN 张量描述符 ---
    // 使用 utils_op.h 提供的 CreateMTensor 包装函数
    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*out, format_);

    // --- 步骤 4: 执行 muDNN 计算 ---
    // muDNN Binary Run 接口通常为: Run(handle, output, input_x, input_y)
    mStatus status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);

    // --- 步骤 5: 检查执行状态 ---
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Minimum execution failed. Status code: ", (int)status));
  }
};

} // namespace musa
} // namespace tensorflow

using namespace tensorflow;

// 注册宏保持不变
#define REGISTER_MUSA_MIN(type)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Minimum").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),  \
      ::tensorflow::musa::MusaMinimumOp<type>);

REGISTER_MUSA_MIN(float);
REGISTER_MUSA_MIN(double);
REGISTER_MUSA_MIN(int32);
REGISTER_MUSA_MIN(int64);
REGISTER_MUSA_MIN(Eigen::half);      // FP16
REGISTER_MUSA_MIN(bfloat16);          // BF16

#undef REGISTER_MUSA_MIN
