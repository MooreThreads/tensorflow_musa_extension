/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

using Var = ::tensorflow::Var;

// 1. MusaVarHandleOp - 创建变量句柄
class MusaVarHandleOp : public OpKernel {
 public:
  explicit MusaVarHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  }
  void Compute(OpKernelContext* ctx) override {
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    ResourceHandle handle = MakeResourceHandle<Var>(ctx, container_, shared_name_);
    out->flat<ResourceHandle>()(0) = handle;
  }
 private:
  string container_;
  string shared_name_;
};

// 2. MusaAssignVariableOp - 变量赋值
template <typename T>
class MusaAssignVariableOp : public OpKernel {
 public:
  explicit MusaAssignVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& value = ctx->input(1);
    
    // 如果运行时请求了输出（如初始化链），转发 Resource Handle
    if (ctx->num_outputs() > 0) {
      ctx->set_output(0, ctx->input(0)); 
    }

    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(ctx, HandleFromInput(ctx, 0), &var,
      [&](Var** ptr) { 
        *ptr = new Var(value.dtype()); 
        return Status::OK(); 
      }));

    mutex_lock lock(*var->mu());
    *var->tensor() = value; // 浅拷贝引用
    var->is_initialized = true;
  }
};



// 3. MusaReadVariableOp 
class MusaReadVariableOp : public OpKernel {
 public:
  explicit MusaReadVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // 【埋点 1】确认进入 Compute
    // std::cerr << ">>>>> [MUSA_READ_LOG] 1. Enter Compute for Node: " << ctx->op_kernel().name() << std::endl;

    core::RefCountPtr<Var> var;
    // 1. 获取 Handle
    const Tensor& handle_tensor = ctx->input(0);
    const ResourceHandle& handle = handle_tensor.flat<ResourceHandle>()(0);
    
    // 【埋点 2】确认 Handle 信息
    // std::cerr << ">>>>> [MUSA_READ_LOG] 2. Handle Name: " << handle.name() << ", Device: " << handle.device() << std::endl;

    // 2. 查找资源
    Status s = LookupResource(ctx, handle, &var);
    if (!s.ok()) {
      // std::cerr << ">>>>> [MUSA_READ_LOG] ❌ 3. LookupResource FAILED: " << s.ToString() << std::endl;
      ctx->CtxFailure(s);
      return;
    }

    tf_shared_lock lock(*var->mu());
    
    // 3. 检查初始化
    if (!var->is_initialized) {
      // std::cerr << ">>>>> [MUSA_READ_LOG] ❌ 4. Variable NOT Initialized!" << std::endl;
      ctx->CtxFailure(errors::FailedPrecondition("Variable not initialized."));
      return;
    }

    // 【埋点 3】确认 Tensor 状态
    const Tensor& t = *var->tensor();
    // std::cerr << ">>>>> [MUSA_READ_LOG] 5. Tensor Ready. DType: " << DataTypeString(t.dtype()) 
    //           << ", Shape: " << t.shape().DebugString() << std::endl;

    // 4. 【核心输出】
    ctx->set_output(0, t);
    
    // 【埋点 4】确认成功结束
    // std::cerr << ">>>>> [MUSA_READ_LOG] 6. set_output(0) SUCCESS. Done." << std::endl;
  }
};

// 注册：保持通用，不带 T 约束
// 注册 ReadVariableOp
REGISTER_KERNEL_BUILDER(Name("ReadVariableOp").Device("MUSA").HostMemory("resource"), MusaReadVariableOp);

// 🌟 增加这一行别名注册，很多版本的 Adam 实际上在找这个名字
REGISTER_KERNEL_BUILDER(Name("ResourceReadVariableOp").Device("MUSA").HostMemory("resource"), MusaReadVariableOp);


// 4. MusaVarIsInitializedOp - 检查变量是否已初始化
class MusaVarIsInitializedOp : public OpKernel {
 public:
  explicit MusaVarIsInitializedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    core::RefCountPtr<Var> var;
    bool is_init = LookupResource(ctx, HandleFromInput(ctx, 0), &var).ok() && var->is_initialized;
    out->flat<bool>()(0) = is_init;
  }
};

// 5. MusaDestroyResourceOp - 销毁资源
class MusaDestroyResourceOp : public OpKernel {
 public:
  explicit MusaDestroyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    DeleteResource(ctx, HandleFromInput(ctx, 0));
  }
};

// --- 注册区 ---


#define REGISTER_MUSA_VAR_MANAGEMENT(T) \
  REGISTER_KERNEL_BUILDER(Name("VarHandleOp").Device("MUSA").HostMemory("resource").TypeConstraint<T>("dtype"), MusaVarHandleOp); \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp").Device("MUSA").HostMemory("resource").TypeConstraint<T>("dtype"), MusaAssignVariableOp<T>); \
  // REGISTER_KERNEL_BUILDER(Name("ReadVariableOp").Device("MUSA").HostMemory("resource").TypeConstraint<T>("dtype"), MusaReadVariableOp<T>);

// 注册常用类型
REGISTER_MUSA_VAR_MANAGEMENT(float);
REGISTER_MUSA_VAR_MANAGEMENT(double); // 增加 double 支持
REGISTER_MUSA_VAR_MANAGEMENT(Eigen::half);
REGISTER_MUSA_VAR_MANAGEMENT(int32);
REGISTER_MUSA_VAR_MANAGEMENT(int64);

// 注册状态与销毁算子
REGISTER_KERNEL_BUILDER(Name("VarIsInitializedOp").Device("MUSA").HostMemory("resource").HostMemory("is_initialized"), MusaVarIsInitializedOp);
REGISTER_KERNEL_BUILDER(Name("DestroyResourceOp").Device("MUSA").HostMemory("resource"), MusaDestroyResourceOp);

} // namespace musa
} // namespace tensorflow



