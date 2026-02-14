#include <cmath>
#include <iostream>
#include <list>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

void ForwardResourceHandle(OpKernelContext* ctx) {
  if (ctx->num_outputs() > 0) {
    if (ctx->input_dtype(0) == DT_RESOURCE) {
      for (int i = 0; i < ctx->num_outputs(); ++i) {
        ctx->set_output(i, ctx->input(i));
      }
    } else {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }
}

template <typename T>
class MusaApplyAdamOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdamOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    Var *var, *m, *v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));
    core::ScopedUnref r1(var), r2(m), r3(v);

    OP_REQUIRES(ctx, var->tensor()->IsInitialized(),
                errors::FailedPrecondition("Var not initialized"));

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temps;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;
    ::musa::dnn::Fill fill_op;

    auto fill_t = [&](float val, const TensorShape& s) {
      temps.emplace_back();
      ctx->allocate_temp(DT_FLOAT, s, &temps.back());
      mTensor t = CreateMTensor(temps.back(), format_);
      fill_op.SetValue(val);
      fill_op.Run(handle, t);
      return t;
    };

    float b1_p = ctx->input(3).scalar<float>()();
    float b2_p = ctx->input(4).scalar<float>()();
    float lr = ctx->input(5).scalar<float>()();
    float b1 = ctx->input(6).scalar<float>()();
    float b2 = ctx->input(7).scalar<float>()();
    float eps = ctx->input(8).scalar<float>()();
    const Tensor& grad = ctx->input(9);

    double alpha = lr * std::sqrt(1.0 - b2_p) / (1.0 - b1_p);

    mTensor t_var = CreateMTensor(*(var->tensor()), format_);
    mTensor t_m = CreateMTensor(*(m->tensor()), format_);
    mTensor t_v = CreateMTensor(*(v->tensor()), format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_m, t_m, fill_t(b1, grad.shape()));
    mTensor t_g_sc = fill_t(1.0f - b1, grad.shape());
    b_op.Run(handle, t_g_sc, t_grad, t_g_sc);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_m, t_m, t_g_sc);

    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_v, t_v, fill_t(b2, grad.shape()));
    temps.emplace_back();
    ctx->allocate_temp(DT_FLOAT, grad.shape(), &temps.back());
    mTensor t_g2 = CreateMTensor(temps.back(), format_);
    b_op.Run(handle, t_g2, t_grad, t_grad);
    mTensor t_g2_sc = fill_t(1.0f - b2, grad.shape());
    b_op.Run(handle, t_g2_sc, t_g2, t_g2_sc);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_v, t_v, t_g2_sc);

    temps.emplace_back();
    ctx->allocate_temp(DT_FLOAT, grad.shape(), &temps.back());
    mTensor t_den = CreateMTensor(temps.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    u_op.Run(handle, t_den, t_v);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_den, t_den, fill_t(eps, grad.shape()));
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_den, t_m, t_den);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_den, t_den, fill_t(alpha, grad.shape()));
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_den);

    ForwardResourceHandle(ctx);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_ADAM(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")      \
                              .Device(DEVICE_MTGPU)      \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("beta1_power") \
                              .HostMemory("beta2_power") \
                              .HostMemory("lr")          \
                              .HostMemory("beta1")       \
                              .HostMemory("beta2")       \
                              .HostMemory("epsilon"),    \
                          MusaApplyAdamOp<T>);

REGISTER_ADAM(float);
REGISTER_ADAM(double);
REGISTER_ADAM(Eigen::half);
REGISTER_ADAM(bfloat16);
REGISTER_ADAM(int64);
REGISTER_ADAM(int32);

}  // namespace musa
}  // namespace tensorflow
