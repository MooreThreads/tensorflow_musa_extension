#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

#include "../array/musa_fill_functor.h"
#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

// Keep Adam-related kernels in one translation unit so similarly shaped helper
// classes do not end up with duplicate names across different .cc files.

Status CopyTensorForUpdate(OpKernelContext* ctx, const Tensor& src,
                           Tensor* dst) {
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  attr.set_nic_compatible(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst, attr));

  if (src.TotalBytes() == 0) {
    return Status::OK();
  }

  // Use musaMemcpyAsync for same-device memory copy
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(dst->data(), src.data(), src.TotalBytes(),
                                    musaMemcpyDeviceToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal("CopyTensorForUpdate: musaMemcpyAsync failed: ",
                            musaGetErrorString(err));
  }

  return Status::OK();
}

Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return Status::OK();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(CopyTensorForUpdate(ctx, *var->tensor(), &copied));
  *var->tensor() = copied;
  return Status::OK();
}

class MutexUnlocker {
 public:
  explicit MutexUnlocker(mutex* mu) : mu_(mu) {}
  ~MutexUnlocker() {
    if (mu_ != nullptr) {
      mu_->unlock();
    }
  }

 private:
  mutex* mu_;
};

template <typename T>
class MusaResourceApplyAdamOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyAdamOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> m;
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));

    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(var->mu());
    add_mutex(m->mu());
    add_mutex(v->mu());
    std::sort(mutexes.begin(), mutexes.end());

    for (mutex* mu : mutexes) {
      mu->lock();
    }
    std::vector<MutexUnlocker> locks;
    locks.reserve(mutexes.size());
    for (mutex* mu : mutexes) {
      locks.emplace_back(mu);
    }

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    m->tensor()->IsInitialized() &&
                    v->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Adam variables (var/m/v) not initialized."));

    // Validate shapes match (Adam requires var, m, v, and grad to have same
    // shape)
    Tensor var_t = *var->tensor();
    Tensor m_t = *m->tensor();
    Tensor v_t = *v->tensor();
    const Tensor& grad = ctx->input(9);

    OP_REQUIRES(
        ctx, var_t.shape().IsSameSize(m_t.shape()),
        errors::InvalidArgument("var and m must have the same shape. var: ",
                                var_t.shape().DebugString(),
                                " m: ", m_t.shape().DebugString()));
    OP_REQUIRES(
        ctx, var_t.shape().IsSameSize(v_t.shape()),
        errors::InvalidArgument("var and v must have the same shape. var: ",
                                var_t.shape().DebugString(),
                                " v: ", v_t.shape().DebugString()));
    OP_REQUIRES(
        ctx, var_t.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad must have the same shape. var: ",
                                var_t.shape().DebugString(),
                                " grad: ", grad.shape().DebugString()));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));

    // Update tensor references after potential copy
    var_t = *var->tensor();
    m_t = *m->tensor();
    v_t = *v->tensor();

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyAdam ", op_name,
                                " failed. Status: ", static_cast<int>(status));
      }
      return Status::OK();
    };

    auto fill_scalar = [&](T val, const TensorShape& shape,
                           mTensor* out) -> Status {
      temp_storage.emplace_back();
      Status alloc_status = ctx->allocate_temp(DataTypeToEnum<T>::value, shape,
                                               &temp_storage.back());
      if (!alloc_status.ok()) {
        return alloc_status;
      }
      *out = CreateMTensor(temp_storage.back(), format_);
      return MusaFillCall(out, val, ctx);
    };

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T beta2_power = ctx->input(4).scalar<T>()();
    const T lr = ctx->input(5).scalar<T>()();
    const T beta1 = ctx->input(6).scalar<T>()();
    const T beta2 = ctx->input(7).scalar<T>()();
    const T epsilon = ctx->input(8).scalar<T>()();
    // grad already declared above for shape validation

    // Calculate bias-corrected learning rate
    // Formula: alpha = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
    // Handle the edge case when beta1_power = 1.0 (initial state) to avoid
    // division by zero
    double alpha_val;
    const double one_minus_beta1_power = 1.0 - static_cast<double>(beta1_power);
    if (std::abs(one_minus_beta1_power) < 1e-10) {
      // Initial iteration: beta1_power ≈ 1.0, use lr as fallback
      // This matches TensorFlow's behavior on the first step
      alpha_val = static_cast<double>(lr);
    } else {
      alpha_val = static_cast<double>(lr) *
                  std::sqrt(1.0 - static_cast<double>(beta2_power)) /
                  one_minus_beta1_power;
    }

    // // Log shapes for debugging
    // LOG(INFO) << "ResourceApplyAdam shapes: var=" <<
    // var_t.shape().DebugString()
    //           << " m=" << m_t.shape().DebugString()
    //           << " v=" << v_t.shape().DebugString()
    //           << " grad=" << grad.shape().DebugString()
    //           << " dtype=" << var_t.dtype();

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_m = CreateMTensor(m_t, format_);
    mTensor t_v = CreateMTensor(v_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    mTensor t_beta1;
    mTensor t_inv_beta1;
    mTensor t_beta2;
    mTensor t_inv_beta2;
    mTensor t_eps;
    mTensor t_alpha;
    // Use var_t.shape() consistently for all scalar fills (var, m, v, grad
    // should all have same shape)
    OP_REQUIRES_OK(ctx, fill_scalar(beta1, var_t.shape(), &t_beta1));
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta1, var_t.shape(),
                                    &t_inv_beta1));
    OP_REQUIRES_OK(ctx, fill_scalar(beta2, var_t.shape(), &t_beta2));
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta2, var_t.shape(),
                                    &t_inv_beta2));
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, v_t.shape(), &t_eps));
    OP_REQUIRES_OK(
        ctx, fill_scalar(static_cast<T>(alpha_val), var_t.shape(), &t_alpha));
    // Step 1: m = beta1 * m + (1 - beta1) * grad
    // Following RMSProp pattern: allow in-place operations for MUL and ADD
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_m, t_m, t_beta1), "MUL beta1"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value, var_t.shape(),
                                      &temp_storage.back()));
    mTensor t_g_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_g_scaled, t_grad, t_inv_beta1),
                             "MUL inv_beta1"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_m, t_m, t_g_scaled), "ADD m"));

    // Step 2: v = beta2 * v + (1 - beta2) * grad^2
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v, t_v, t_beta2), "MUL beta2"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value, var_t.shape(),
                                      &temp_storage.back()));
    mTensor t_g2 = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_g2, t_grad, t_grad),
                                        "MUL grad_sq"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value, var_t.shape(),
                                      &temp_storage.back()));
    mTensor t_g2_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_g2_scaled, t_g2, t_inv_beta2),
                             "MUL inv_beta2"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v, t_v, t_g2_scaled), "ADD v"));

    // Step 3: sqrt(v) + epsilon
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_sqrt_v = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(ctx,
                   require_success(u_op.Run(handle, t_sqrt_v, t_v), "SQRT v"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_v_plus_eps = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v_plus_eps, t_sqrt_v, t_eps),
                             "ADD epsilon"));

    // Step 4: update = m / denom
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value, var_t.shape(),
                                      &temp_storage.back()));
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_update, t_m, t_v_plus_eps),
                             "DIV update"));

    // Step 5: update = update * alpha
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value, var_t.shape(),
                                      &temp_storage.back()));
    mTensor t_update_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_update_scaled,
                                                 t_update, t_alpha),
                                        "MUL alpha"));

    // Step 6: var = var - update_scaled (in-place SUB like RMSProp)
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_var, t_var, t_update_scaled),
                             "SUB var"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyAdam: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

template <typename T>
class MusaApplyAdamKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdamKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  // ApplyAdam is computationally intensive (multiple element-wise ops).
  bool IsExpensive() override { return true; }

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

    const Tensor& grad = ctx->input(9);

    // Validate shapes match
    OP_REQUIRES(
        ctx, var_t->shape().IsSameSize(m_t->shape()),
        errors::InvalidArgument("var and m must have the same shape. var: ",
                                var_t->shape().DebugString(),
                                " m: ", m_t->shape().DebugString()));
    OP_REQUIRES(
        ctx, var_t->shape().IsSameSize(v_t->shape()),
        errors::InvalidArgument("var and v must have the same shape. var: ",
                                var_t->shape().DebugString(),
                                " v: ", v_t->shape().DebugString()));
    OP_REQUIRES(
        ctx, var_t->shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad must have the same shape. var: ",
                                var_t->shape().DebugString(),
                                " grad: ", grad.shape().DebugString()));

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T beta2_power = ctx->input(4).scalar<T>()();
    const T lr = ctx->input(5).scalar<T>()();
    const T beta1 = ctx->input(6).scalar<T>()();
    const T beta2 = ctx->input(7).scalar<T>()();
    const T epsilon = ctx->input(8).scalar<T>()();
    // grad already declared above for shape validation

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
      ::musa::dnn::Fill fill_op;
      fill_op.SetValue(static_cast<float>(val));
      return fill_op.Run(handle, *out);
    };

    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    // Calculate bias-corrected learning rate
    // Handle the edge case when beta1_power = 1.0 (initial state) to avoid
    // division by zero
    double alpha_val;
    const double one_minus_beta1_power = 1.0 - static_cast<double>(beta1_power);
    if (std::abs(one_minus_beta1_power) < 1e-10) {
      // Initial iteration: beta1_power ≈ 1.0, use lr as fallback
      alpha_val = static_cast<double>(lr);
    } else {
      alpha_val = static_cast<double>(lr) *
                  std::sqrt(1.0 - static_cast<double>(beta2_power)) /
                  one_minus_beta1_power;
    }

    mTensor t_beta1;
    mTensor t_inv_beta1;
    mTensor t_beta2;
    mTensor t_inv_beta2;
    mTensor t_eps;
    mTensor t_alpha;
    // Use var_t->shape() consistently for all scalar fills (var, m, v, grad
    // should all have same shape)
    fill_scalar(beta1, var_t->shape(), &t_beta1);
    fill_scalar(static_cast<T>(1.0) - beta1, var_t->shape(), &t_inv_beta1);
    fill_scalar(beta2, var_t->shape(), &t_beta2);
    fill_scalar(static_cast<T>(1.0) - beta2, var_t->shape(), &t_inv_beta2);
    fill_scalar(epsilon, v_t->shape(), &t_eps);
    fill_scalar(static_cast<T>(alpha_val), var_t->shape(), &t_alpha);

    // Step 1: m = beta1 * m + (1 - beta1) * grad
    // Avoid all in-place operations for safety
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_m_new = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_m_new, t_m, t_beta1);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_grad_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_scaled, t_grad, t_inv_beta1);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_m, t_m_new, t_grad_scaled);

    // Step 2: v = beta2 * v + (1 - beta2) * grad^2
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_v_new = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_v_new, t_v, t_beta2);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq, t_grad, t_grad);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_grad_sq_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq_scaled, t_grad_sq, t_inv_beta2);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_v, t_v_new, t_grad_sq_scaled);

    // Step 3: sqrt(v) + epsilon
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t->shape(),
                       &temp_storage.back());
    mTensor t_sqrt_v = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    u_op.Run(handle, t_sqrt_v, t_v);

    // sqrt_v + epsilon (allocate new output)
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t->shape(),
                       &temp_storage.back());
    mTensor t_denom = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_denom, t_sqrt_v, t_eps);

    // Step 4: update = m / denom
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_update, t_m, t_denom);

    // Step 5: update = update * alpha
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, var_t->shape(),
                       &temp_storage.back());
    mTensor t_update_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_update_scaled, t_update, t_alpha);

    // Step 6: var = var - update_scaled
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_update_scaled);

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

#define REGISTER_RESOURCE_ADAM(T)                        \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")      \
                              .Device(DEVICE_MTGPU)      \
                              .HostMemory("var")         \
                              .HostMemory("m")           \
                              .HostMemory("v")           \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("beta1_power") \
                              .HostMemory("beta2_power") \
                              .HostMemory("lr")          \
                              .HostMemory("beta1")       \
                              .HostMemory("beta2")       \
                              .HostMemory("epsilon"),    \
                          MusaResourceApplyAdamOp<T>);

#define REGISTER_APPLY_ADAM(T)                           \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdam")              \
                              .Device(DEVICE_MTGPU)      \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("beta1_power") \
                              .HostMemory("beta2_power") \
                              .HostMemory("lr")          \
                              .HostMemory("beta1")       \
                              .HostMemory("beta2")       \
                              .HostMemory("epsilon"),    \
                          MusaApplyAdamKernelOp<T>);

REGISTER_RESOURCE_ADAM(float);
REGISTER_RESOURCE_ADAM(double);
REGISTER_RESOURCE_ADAM(Eigen::half);
REGISTER_RESOURCE_ADAM(bfloat16);
REGISTER_RESOURCE_ADAM(int64);
REGISTER_RESOURCE_ADAM(int32);

REGISTER_APPLY_ADAM(float);
REGISTER_APPLY_ADAM(double);
REGISTER_APPLY_ADAM(Eigen::half);
REGISTER_APPLY_ADAM(bfloat16);

#undef REGISTER_RESOURCE_ADAM
#undef REGISTER_APPLY_ADAM

}  // namespace musa
}  // namespace tensorflow
