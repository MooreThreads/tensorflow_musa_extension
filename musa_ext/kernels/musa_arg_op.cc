/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "tensorflow/core/framework/function.h"  // 👈 必须包含这个头文件
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_handle.h"

namespace tensorflow {
namespace musa {

// =================================================================
// 1. MusaArgOp: 从 CallFrame 中提取函数参数 (适配双指针语法)
// =================================================================
class MusaArgOp : public OpKernel {
 public:
  explicit MusaArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto* frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, 
                errors::Internal("MUSA _Arg: No call frame found."));

    const Tensor* val_ptr = nullptr;
   
    Status s = frame->GetArg(index_, &val_ptr);
    OP_REQUIRES_OK(ctx, s);
    OP_REQUIRES(ctx, val_ptr != nullptr, 
                errors::Internal("MUSA _Arg: Retrieved null tensor pointer."));

    // 将拿到的 Tensor 设置为输出
    ctx->set_output(0, *val_ptr);
  }

 private:
  int index_;
};

// =================================================================
// 2. MusaRetvalOp: 将结果存入 CallFrame 返回值
// =================================================================
class MusaRetvalOp : public OpKernel {
 public:
  explicit MusaRetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    auto* frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, 
                errors::Internal("MUSA _Retval: No call frame found."));

    // SetRetval 的签名是 (int, const Tensor&)，直接传即可
    Status s = frame->SetRetval(index_, input);
    OP_REQUIRES_OK(ctx, s);
  }

 private:
  int index_;
};

// =================================================================
// 3. 注册区
// =================================================================

#define REGISTER_MUSA_FUN_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("_Arg").Device("MUSA").TypeConstraint<type>("T"), MusaArgOp); \
  REGISTER_KERNEL_BUILDER(Name("_Retval").Device("MUSA").TypeConstraint<type>("T"), MusaRetvalOp);

REGISTER_MUSA_FUN_KERNELS(float);
REGISTER_MUSA_FUN_KERNELS(double);
REGISTER_MUSA_FUN_KERNELS(Eigen::half);
REGISTER_MUSA_FUN_KERNELS(int32);
REGISTER_MUSA_FUN_KERNELS(int64);
REGISTER_MUSA_FUN_KERNELS(bool);

// ResourceHandle 的注册
REGISTER_KERNEL_BUILDER(Name("_Arg").Device("MUSA")
                            .HostMemory("output")
                            .TypeConstraint<ResourceHandle>("T"), MusaArgOp);

REGISTER_KERNEL_BUILDER(Name("_Retval").Device("MUSA")
                            .HostMemory("input")
                            .TypeConstraint<ResourceHandle>("T"), MusaRetvalOp);

}  // namespace musa
}  // namespace tensorflow
