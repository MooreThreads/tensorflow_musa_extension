#include <cmath>
#include <list>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaApplyAdamOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdamOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    Var* var = nullptr;
    Var* m = nullptr;
    Var* v = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));
    core::ScopedUnref var_unref(var), m_unref(m), v_unref(v);

    Tensor* var_t = var->tensor();
    Tensor* m_t = m->tensor();
    Tensor* v_t = v->tensor();

    OP_REQUIRES(
        ctx,
        var_t->IsInitialized() && m_t->IsInitialized() && v_t->IsInitialized(),
        errors::FailedPrecondition(
            "Adam variables (var/m/v) not initialized."));

    const T beta1_p = ctx->input(3).scalar<T>()();
    const T beta2_p = ctx->input(4).scalar<T>()();
    const T lr = ctx->input(5).scalar<T>()();
    const T beta1 = ctx->input(6).scalar<T>()();
    const T beta2 = ctx->input(7).scalar<T>()();
    const T epsilon = ctx->input(8).scalar<T>()();
    const Tensor& grad = ctx->input(9);

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var = CreateMTensor(*var_t, format_);
    mTensor t_m = CreateMTensor(*m_t, format_);
    mTensor t_v = CreateMTensor(*v_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    auto fill_scalar = [&](T val, const TensorShape& shape, mTensor* out) {
      temp_storage.emplace_back();
      ctx->allocate_temp(DataTypeToEnum<T>::value, shape, &temp_storage.back());
      *out = CreateMTensor(temp_storage.back(), format_);
      ::musa::dnn::Fill f;
      f.SetValue(static_cast<float>(val));
      return f.Run(handle, *out);
    };

    ::musa::dnn::Binary b_op;

    ::musa::dnn::Unary u_op;

    double alpha_val = static_cast<double>(lr) *
                       std::sqrt(1.0 - static_cast<double>(beta2_p)) /
                       (1.0 - static_cast<double>(beta1_p));

    mTensor t_b1, t_inv_b1;
    fill_scalar(beta1, m_t->shape(), &t_b1);
    fill_scalar(static_cast<T>(1.0) - beta1, grad.shape(), &t_inv_b1);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_m, t_m, t_b1);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_g_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_g_scaled, t_grad, t_inv_b1);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_m, t_m, t_g_scaled);

    mTensor t_b2, t_inv_b2;
    fill_scalar(beta2, v_t->shape(), &t_b2);
    fill_scalar(static_cast<T>(1.0) - beta2, grad.shape(), &t_inv_b2);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_v, t_v, t_b2);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_g2 = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_g2, t_grad, t_grad);
    ;
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_g2_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_g2_scaled, t_g2, t_inv_b2);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_v, t_v, t_g2_scaled);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t->shape(),
                       &temp_storage.back());
    mTensor t_sqrt_v = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    u_op.Run(handle, t_sqrt_v, t_v);
    mTensor t_eps;
    fill_scalar(epsilon, v_t->shape(), &t_eps);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_sqrt_v, t_sqrt_v, t_eps);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_sqrt_v, t_m, t_sqrt_v);
    mTensor t_alpha;
    fill_scalar(static_cast<T>(alpha_val), var_t->shape(), &t_alpha);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_sqrt_v, t_sqrt_v, t_alpha);
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_sqrt_v);

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      for (int i = 0; i < ctx->num_outputs(); ++i) {
        ctx->set_output(i, ctx->input(i));
      }
    }
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_MUSA(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdam")              \
                              .Device(DEVICE_MTGPU)      \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("beta1_power") \
                              .HostMemory("beta2_power") \
                              .HostMemory("lr")          \
                              .HostMemory("beta1")       \
                              .HostMemory("beta2")       \
                              .HostMemory("epsilon"),    \
                          MusaApplyAdamOp<T>);

REGISTER_MUSA(float);
REGISTER_MUSA(double);
REGISTER_MUSA(Eigen::half);
REGISTER_MUSA(bfloat16);

}  // namespace musa
}  // namespace tensorflow