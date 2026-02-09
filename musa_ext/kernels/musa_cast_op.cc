/* Copyright @2020-2026 Moore Threads Technology Co. All rights reserved. */

#include "utils_op.h" 
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

class MusaCastOp : public MusaOpKernel {
 public:
  explicit MusaCastOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &output));

    if (inp.NumElements() == 0) return;

    // 1. 准备 muDNN 张量描述符

    auto in_mt = CreateMTensor(inp);
    auto out_mt = CreateMTensor(*output);

    mHandle& h = GetHandleByCtx(ctx);
    ::musa::dnn::Unary op;

    // 2. 设置 Cast 模式

    mStatus m_status;
    if (external_src_dtype_ == external_dst_dtype_) {
        m_status = op.SetMode(::musa::dnn::Unary::Mode::IDENTITY);
    } else {
        m_status = op.SetMode(::musa::dnn::Unary::Mode::CAST);
    }
    
    OP_REQUIRES(ctx, m_status == mStatus::SUCCESS,
                errors::Internal("muDNN Unary SetMode failed in Cast"));

    // 3. 执行 Cast

    m_status = op.Run(h, out_mt, in_mt);

    if (m_status != mStatus::SUCCESS) {
        LOG(ERROR) << "MUSA Cast Run failed! Src: " << DataTypeString(external_src_dtype_)
                   << " -> Dst: " << DataTypeString(external_dst_dtype_)
                   << " | Status: " << (int)m_status;
        
        ctx->SetStatus(errors::Internal("MUSA Cast Run failed. Status code: ", (int)m_status));
        return;
    }
  }

 private:
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
};

// --- 广泛的类型注册 ---
#define REGISTER_CAST_MUSA(SrcT, DstT) \
  REGISTER_KERNEL_BUILDER(Name("Cast") \
                          .Device(DEVICE_MTGPU) \
                          .TypeConstraint<SrcT>("SrcT") \
                          .TypeConstraint<DstT>("DstT"), \
                          MusaCastOp);

// 数值类型互转
REGISTER_CAST_MUSA(float, int32);
REGISTER_CAST_MUSA(int32, float);
REGISTER_CAST_MUSA(float, int64);
REGISTER_CAST_MUSA(int64, float);
REGISTER_CAST_MUSA(int32, int64);
REGISTER_CAST_MUSA(int64, int32);
REGISTER_CAST_MUSA(float, Eigen::half);
REGISTER_CAST_MUSA(Eigen::half, float);
REGISTER_CAST_MUSA(float, bfloat16);
REGISTER_CAST_MUSA(bfloat16, float);

REGISTER_CAST_MUSA(bool, float);
REGISTER_CAST_MUSA(bool, int32);
REGISTER_CAST_MUSA(int32, bool);

REGISTER_CAST_MUSA(float, float);
REGISTER_CAST_MUSA(int32, int32);
REGISTER_CAST_MUSA(int64, int64);

} // namespace musa
} // namespace tensorflow


