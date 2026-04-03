/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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

// Keep AdaMax-related kernels: reuse helper functions from musa_applyadam_op.cc
// These functions are already defined in musa_applyadam_op.cc

// Helper function declarations (defined in musa_applyadam_op.cc)
extern Status CopyTensorForUpdate(OpKernelContext* ctx, const Tensor& src,
                                   Tensor* dst);
extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

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

// Validate AdaMax hyperparameters are scalars
inline Status ValidateAdaMaxHyperParams(OpKernelContext* ctx) {
  const auto require_scalar = [&](int index, const char* name) -> Status {
    const Tensor& t = ctx->input(index);
    if (t.NumElements() != 1) {
      return errors::InvalidArgument(name, " must be a scalar, got shape ",
                                     t.shape().DebugString());
    }
    return Status::OK();
  };

  // AdaMax uses: var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad
  // So hyperparameters are at indices 3-7
  TF_RETURN_IF_ERROR(require_scalar(3, "beta1_power"));
  TF_RETURN_IF_ERROR(require_scalar(4, "lr"));
  TF_RETURN_IF_ERROR(require_scalar(5, "beta1"));
  TF_RETURN_IF_ERROR(require_scalar(6, "beta2"));
  TF_RETURN_IF_ERROR(require_scalar(7, "epsilon"));
  return Status::OK();
}

// Validate that var, m, v, and grad have the same shape
inline Status ValidateAdaMaxShapes(const Tensor& var_t, const Tensor& m_t,
                                   const Tensor& v_t, const Tensor& grad_t) {
  if (!var_t.shape().IsSameSize(m_t.shape())) {
    return errors::InvalidArgument(
        "var and m must have the same shape. var: ",
        var_t.shape().DebugString(), " m: ", m_t.shape().DebugString());
  }

  if (!var_t.shape().IsSameSize(v_t.shape())) {
    return errors::InvalidArgument(
        "var and v must have the same shape. var: ",
        var_t.shape().DebugString(), " v: ", v_t.shape().DebugString());
  }

  if (!var_t.shape().IsSameSize(grad_t.shape())) {
    return errors::InvalidArgument(
        "var and grad must have the same shape. var: ",
        var_t.shape().DebugString(), " grad: ", grad_t.shape().DebugString());
  }

  return Status::OK();
}

template <typename T>
class MusaResourceApplyAdaMaxOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyAdaMaxOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
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
                    "AdaMax variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));

    Tensor var_t = *var->tensor();
    Tensor m_t = *m->tensor();
    Tensor v_t = *v->tensor();

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyAdaMax ", op_name,
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
    const T lr = ctx->input(4).scalar<T>()();
    const T beta1 = ctx->input(5).scalar<T>()();
    const T beta2 = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();
    const Tensor& grad = ctx->input(8);

    // AdaMax algorithm:
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t = max(beta2 * v_{t-1}, |g_t|)
    // lr_t = lr / (1 - beta1^t)
    // var_t = var_{t-1} - lr_t * m_t / (v_t + epsilon)

    // Calculate learning rate decay
    const double denom = 1.0 - static_cast<double>(beta1_power);
    if (denom == 0.0) {
      ctx->CtxFailure(errors::InvalidArgument("beta1_power must not be 1."));
      return;
    }
    const T lr_t = static_cast<T>(static_cast<double>(lr) / denom);

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_m = CreateMTensor(m_t, format_);
    mTensor t_v = CreateMTensor(v_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    // Step 1: Update m: m = beta1 * m + (1 - beta1) * grad
    mTensor t_beta1;
    mTensor t_inv_beta1;
    OP_REQUIRES_OK(ctx, fill_scalar(beta1, m_t.shape(), &t_beta1));
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta1, grad.shape(),
                                    &t_inv_beta1));

    // m *= beta1
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_m, t_m, t_beta1), "MUL beta1"));

    // grad * (1-beta1)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_g_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_g_scaled, t_grad, t_inv_beta1),
                             "MUL inv_beta1"));

    // m += grad * (1-beta1)
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_m, t_m, t_g_scaled), "ADD m"));

    // Step 2: Update v: v = max(beta2 * v, |grad|)
    // Note: MuDNN doesn't have MAX, so we use: max(a,b) = (a+b+|a-b|)/2

    // beta2 * v
    mTensor t_beta2;
    OP_REQUIRES_OK(ctx, fill_scalar(beta2, v_t.shape(), &t_beta2));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_v_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v_scaled, t_v, t_beta2),
                             "MUL beta2"));

    // |grad|
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_abs_grad = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::ABS);
    OP_REQUIRES_OK(
        ctx, require_success(u_op.Run(handle, t_abs_grad, t_grad), "ABS grad"));

    // diff = beta2*v - |grad|
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_diff = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_diff, t_v_scaled, t_abs_grad),
                             "SUB diff"));

    // |diff|
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_abs_diff = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::ABS);
    OP_REQUIRES_OK(
        ctx, require_success(u_op.Run(handle, t_abs_diff, t_diff), "ABS diff"));

    // sum = beta2*v + |grad| + |diff|
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_sum = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_sum, t_v_scaled, t_abs_grad),
                             "ADD sum1"));
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_sum, t_sum, t_abs_diff),
                             "ADD sum2"));

    // max = sum / 2
    mTensor t_half;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(0.5), v_t.shape(), &t_half));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v, t_sum, t_half), "MUL half"));

    // Step 3: Update var: var = var - lr_t * m / (v + epsilon)

    // denominator = v + epsilon
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_den = CreateMTensor(temp_storage.back(), format_);
    mTensor t_eps;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, v_t.shape(), &t_eps));
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_den, t_v, t_eps),
                                        "ADD epsilon"));

    // m / (v + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           m_t.shape(), &temp_storage.back()));
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_update, t_m, t_den), "DIV"));

    // lr_t * m / (v + epsilon)
    mTensor t_lr_t;
    OP_REQUIRES_OK(ctx, fill_scalar(lr_t, m_t.shape(), &t_lr_t));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_update, t_update, t_lr_t),
                             "MUL lr_t"));

    // var -= lr_t * m / (v + epsilon)
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_var, t_var, t_update), "SUB var"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyAdaMax: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class MusaApplyAdaMaxKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdaMaxKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  // ApplyAdaMax is computationally intensive (multiple element-wise ops).
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(ctx->input_ref_mutex(0));
    add_mutex(ctx->input_ref_mutex(1));
    add_mutex(ctx->input_ref_mutex(2));
    std::sort(mutexes.begin(), mutexes.end());

    std::vector<MutexUnlocker> locks;
    if (use_exclusive_lock_) {
      for (mutex* mu : mutexes) {
        mu->lock();
      }
      locks.reserve(mutexes.size());
      for (mutex* mu : mutexes) {
        locks.emplace_back(mu);
      }
    }

    OP_REQUIRES_OK(ctx, ValidateAdaMaxHyperParams(ctx));

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }

    Tensor var_t = ctx->mutable_input(0, true);
    Tensor m_t = ctx->mutable_input(1, true);
    Tensor v_t = ctx->mutable_input(2, true);
    const Tensor& grad = ctx->input(8);

    // Unified initialization check (consistent with Resource version)
    OP_REQUIRES(ctx,
                var_t.IsInitialized() && m_t.IsInitialized() &&
                    v_t.IsInitialized(),
                errors::FailedPrecondition(
                    "AdaMax variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, ValidateAdaMaxShapes(var_t, m_t, v_t, grad));

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T lr = ctx->input(4).scalar<T>()();
    const T beta1 = ctx->input(5).scalar<T>()();
    const T beta2 = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto fill_scalar = [&](T val, const TensorShape& shape, mTensor* out) {
      temp_storage.emplace_back();
      ctx->allocate_temp(DataTypeToEnum<T>::value, shape, &temp_storage.back());
      *out = CreateMTensor(temp_storage.back(), format_);
      ::musa::dnn::Fill fill_op;
      fill_op.SetValue(static_cast<float>(val));
      return fill_op.Run(handle, *out);
    };

    // Calculate learning rate decay
    const double denom = 1.0 - static_cast<double>(beta1_power);
    if (denom == 0.0) {
      ctx->CtxFailure(errors::InvalidArgument("beta1_power must not be 1."));
      return;
    }
    const T lr_t = static_cast<T>(static_cast<double>(lr) / denom);

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_m = CreateMTensor(m_t, format_);
    mTensor t_v = CreateMTensor(v_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    // Step 1: Update m: m = beta1 * m + (1 - beta1) * grad
    mTensor t_beta1;
    mTensor t_inv_beta1;
    fill_scalar(beta1, m_t.shape(), &t_beta1);
    fill_scalar(static_cast<T>(1.0) - beta1, grad.shape(), &t_inv_beta1);

    // m *= beta1
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_m, t_m, t_beta1);

    // grad * (1-beta1)
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_g_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_g_scaled, t_grad, t_inv_beta1);

    // m += grad * (1-beta1)
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_m, t_m, t_g_scaled);

    // Step 2: Update v: v = max(beta2 * v, |grad|)
    // max(a,b) = (a+b+|a-b|)/2

    // beta2 * v
    mTensor t_beta2;
    fill_scalar(beta2, v_t.shape(), &t_beta2);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t.shape(),
                       &temp_storage.back());
    mTensor t_v_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_v_scaled, t_v, t_beta2);

    // |grad|
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_abs_grad = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::ABS);
    u_op.Run(handle, t_abs_grad, t_grad);

    // diff = beta2*v - |grad|
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t.shape(),
                       &temp_storage.back());
    mTensor t_diff = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_diff, t_v_scaled, t_abs_grad);

    // |diff|
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t.shape(),
                       &temp_storage.back());
    mTensor t_abs_diff = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::ABS);
    u_op.Run(handle, t_abs_diff, t_diff);

    // sum = beta2*v + |grad| + |diff|
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t.shape(),
                       &temp_storage.back());
    mTensor t_sum = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_sum, t_v_scaled, t_abs_grad);
    b_op.Run(handle, t_sum, t_sum, t_abs_diff);

    // max = sum / 2
    mTensor t_half;
    fill_scalar(static_cast<T>(0.5), v_t.shape(), &t_half);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_v, t_sum, t_half);

    // Step 3: Update var: var = var - lr_t * m / (v + epsilon)

    // denominator = v + epsilon
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, v_t.shape(),
                       &temp_storage.back());
    mTensor t_den = CreateMTensor(temp_storage.back(), format_);
    mTensor t_eps;
    fill_scalar(epsilon, v_t.shape(), &t_eps);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_den, t_v, t_eps);

    // m / (v + epsilon)
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, m_t.shape(),
                       &temp_storage.back());
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_update, t_m, t_den);

    // lr_t * m / (v + epsilon)
    mTensor t_lr_t;
    fill_scalar(lr_t, m_t.shape(), &t_lr_t);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_update, t_update, t_lr_t);

    // var -= lr_t * m / (v + epsilon)
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_update);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ApplyAdaMax: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_RESOURCE_ADAMAX(T)                       \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdaMax")     \
                              .Device(DEVICE_MTGPU)       \
                              .HostMemory("var")          \
                              .HostMemory("m")            \
                              .HostMemory("v")            \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("beta1_power")  \
                              .HostMemory("lr")           \
                              .HostMemory("beta1")        \
                              .HostMemory("beta2")        \
                              .HostMemory("epsilon"),     \
                          MusaResourceApplyAdaMaxOp<T>);

#define REGISTER_APPLY_ADAMAX(T)                          \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdaMax")             \
                              .Device(DEVICE_MTGPU)       \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("beta1_power")  \
                              .HostMemory("lr")           \
                              .HostMemory("beta1")        \
                              .HostMemory("beta2")        \
                              .HostMemory("epsilon"),     \
                          MusaApplyAdaMaxKernelOp<T>);

// Note: We don't register double/float64 as MuDNN Binary operations may not
// support it. TensorFlow will automatically fallback to CPU for float64.
REGISTER_RESOURCE_ADAMAX(float);
REGISTER_RESOURCE_ADAMAX(Eigen::half);
REGISTER_RESOURCE_ADAMAX(bfloat16);

REGISTER_APPLY_ADAMAX(float);
REGISTER_APPLY_ADAMAX(Eigen::half);
REGISTER_APPLY_ADAMAX(bfloat16);

#undef REGISTER_RESOURCE_ADAMAX
#undef REGISTER_APPLY_ADAMAX

}  // namespace musa
}  // namespace tensorflow