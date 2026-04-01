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

// RMSProp-related kernels following the same pattern as Adam implementation.

// Helper function to copy tensor for update
Status CopyTensorForUpdateRMSProp(OpKernelContext* ctx, const Tensor& src,
                                  Tensor* dst) {
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  attr.set_nic_compatible(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst, attr));

  if (src.TotalBytes() == 0) {
    return Status::OK();
  }

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(dst->data(), src.data(), src.TotalBytes(),
                                    musaMemcpyDeviceToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal("CopyTensorForUpdateRMSProp: musaMemcpyAsync failed: ",
                            musaGetErrorString(err));
  }

  return Status::OK();
}

// Helper function to prepare tensor for MUSA update
Status PrepareTensorForMusaUpdateRMSProp(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return Status::OK();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(CopyTensorForUpdateRMSProp(ctx, *var->tensor(), &copied));
  *var->tensor() = copied;
  return Status::OK();
}

// Mutex unlocker helper class
class MutexUnlockerRMSProp {
 public:
  explicit MutexUnlockerRMSProp(mutex* mu) : mu_(mu) {}
  ~MutexUnlockerRMSProp() {
    if (mu_ != nullptr) {
      mu_->unlock();
    }
  }

 private:
  mutex* mu_;
};

// ResourceApplyRMSProp Op using resource variables
// RMSProp update formulas:
//   ms <- rho * ms_{t-1} + (1-rho) * grad^2
//   mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
//   var <- var - mom
template <typename T>
class MusaResourceApplyRMSPropOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyRMSPropOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> ms;
    core::RefCountPtr<Var> mom;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &ms));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &mom));

    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(var->mu());
    add_mutex(ms->mu());
    add_mutex(mom->mu());
    std::sort(mutexes.begin(), mutexes.end());

    for (mutex* mu : mutexes) {
      mu->lock();
    }
    std::vector<MutexUnlockerRMSProp> locks;
    locks.reserve(mutexes.size());
    for (mutex* mu : mutexes) {
      locks.emplace_back(mu);
    }

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    ms->tensor()->IsInitialized() &&
                    mom->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "RMSProp variables (var/ms/mom) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, ms.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, mom.get()));

    Tensor var_t = *var->tensor();
    Tensor ms_t = *ms->tensor();
    Tensor mom_t = *mom->tensor();

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyRMSProp ", op_name,
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

    // Inputs: lr (3), rho (4), momentum (5), epsilon (6), grad (7)
    const T lr = ctx->input(3).scalar<T>()();
    const T rho = ctx->input(4).scalar<T>()();
    const T momentum = ctx->input(5).scalar<T>()();
    const T epsilon = ctx->input(6).scalar<T>()();
    const Tensor& grad = ctx->input(7);

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_ms = CreateMTensor(ms_t, format_);
    mTensor t_mom = CreateMTensor(mom_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    // Step 1: ms <- rho * ms + (1-rho) * grad^2
    // First: ms <- rho * ms
    mTensor t_rho;
    OP_REQUIRES_OK(ctx, fill_scalar(rho, ms_t.shape(), &t_rho));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_ms, t_ms, t_rho),
                                        "MUL rho_ms"));

    // Second: grad^2
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_sq, t_grad, t_grad),
                                        "MUL grad_sq"));

    // Third: (1-rho) * grad^2
    mTensor t_one_minus_rho;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - rho, grad.shape(),
                                    &t_one_minus_rho));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_sq_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_grad_sq_scaled, t_grad_sq,
                                            t_one_minus_rho),
                                   "MUL grad_sq_scaled"));

    // Fourth: ms <- ms + (1-rho) * grad^2
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_ms, t_ms, t_grad_sq_scaled),
                                        "ADD ms"));

    // Step 2: mom <- momentum * mom + lr * grad / sqrt(ms + epsilon)
    // First: momentum * mom
    mTensor t_momentum;
    OP_REQUIRES_OK(ctx, fill_scalar(momentum, mom_t.shape(), &t_momentum));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mom, t_mom, t_momentum),
                                        "MUL momentum_mom"));

    // Second: ms + epsilon (must add epsilon BEFORE sqrt for correct formula)
    mTensor t_eps;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, ms_t.shape(), &t_eps));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           ms_t.shape(), &temp_storage.back()));
    mTensor t_ms_plus_eps = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_ms_plus_eps, t_ms, t_eps),
                                        "ADD ms_epsilon"));

    // Third: sqrt(ms + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           ms_t.shape(), &temp_storage.back()));
    mTensor t_sqrt_ms = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(ctx, require_success(u_op.Run(handle, t_sqrt_ms, t_ms_plus_eps),
                                        "SQRT ms_plus_eps"));

    // Fourth: grad / sqrt(ms + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_div, t_grad, t_sqrt_ms),
                                        "DIV grad_sqrt"));

    // Fifth: lr * grad / sqrt(ms + epsilon)
    mTensor t_lr;
    OP_REQUIRES_OK(ctx, fill_scalar(lr, grad.shape(), &t_lr));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_lr_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_lr_grad_div, t_grad_div, t_lr),
                                        "MUL lr_grad"));

    // Sixth: mom <- momentum * mom + lr * grad / sqrt(ms + epsilon)
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mom, t_mom, t_lr_grad_div),
                                        "ADD mom"));

    // Step 3: var <- var - mom
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_var, t_var, t_mom),
                                        "SUB var"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyRMSProp: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

// ApplyRMSProp Op using Ref tensors (non-resource version)
template <typename T>
class MusaApplyRMSPropKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyRMSPropKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    Var* var = nullptr;
    Var* ms = nullptr;
    Var* mom = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &ms));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &mom));
    core::ScopedUnref var_unref(var), ms_unref(ms), mom_unref(mom);

    Tensor* var_t = var->tensor();
    Tensor* ms_t = ms->tensor();
    Tensor* mom_t = mom->tensor();

    OP_REQUIRES(ctx,
                var_t->IsInitialized() && ms_t->IsInitialized() &&
                    mom_t->IsInitialized(),
                errors::FailedPrecondition(
                    "RMSProp variables (var/ms/mom) not initialized."));

    // Inputs: lr (3), rho (4), momentum (5), epsilon (6), grad (7)
    const T lr = ctx->input(3).scalar<T>()();
    const T rho = ctx->input(4).scalar<T>()();
    const T momentum = ctx->input(5).scalar<T>()();
    const T epsilon = ctx->input(6).scalar<T>()();
    const Tensor& grad = ctx->input(7);

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var = CreateMTensor(*var_t, format_);
    mTensor t_ms = CreateMTensor(*ms_t, format_);
    mTensor t_mom = CreateMTensor(*mom_t, format_);
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

    // Step 1: ms <- rho * ms + (1-rho) * grad^2
    mTensor t_rho;
    fill_scalar(rho, ms_t->shape(), &t_rho);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_ms, t_ms, t_rho);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq, t_grad, t_grad);

    mTensor t_one_minus_rho;
    fill_scalar(static_cast<T>(1.0) - rho, grad.shape(), &t_one_minus_rho);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_sq_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq_scaled, t_grad_sq, t_one_minus_rho);

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_ms, t_ms, t_grad_sq_scaled);

    // Step 2: mom <- momentum * mom + lr * grad / sqrt(ms + epsilon)
    mTensor t_momentum;
    fill_scalar(momentum, mom_t->shape(), &t_momentum);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_mom, t_mom, t_momentum);

    // ms + epsilon (add epsilon BEFORE sqrt for correct formula)
    mTensor t_eps;
    fill_scalar(epsilon, ms_t->shape(), &t_eps);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, ms_t->shape(),
                       &temp_storage.back());
    mTensor t_ms_plus_eps = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_ms_plus_eps, t_ms, t_eps);

    // sqrt(ms + epsilon)
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, ms_t->shape(),
                       &temp_storage.back());
    mTensor t_sqrt_ms = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    u_op.Run(handle, t_sqrt_ms, t_ms_plus_eps);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_grad_div, t_grad, t_sqrt_ms);

    mTensor t_lr;
    fill_scalar(lr, grad.shape(), &t_lr);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_lr_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_lr_grad_div, t_grad_div, t_lr);

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_mom, t_mom, t_lr_grad_div);

    // Step 3: var <- var - mom
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_mom);

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
};

#define REGISTER_RESOURCE_RMSPROP(T)                        \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyRMSProp")      \
                              .Device(DEVICE_MTGPU)         \
                              .HostMemory("var")            \
                              .HostMemory("ms")             \
                              .HostMemory("mom")            \
                              .TypeConstraint<T>("T")       \
                              .HostMemory("lr")             \
                              .HostMemory("rho")            \
                              .HostMemory("momentum")       \
                              .HostMemory("epsilon"),       \
                          MusaResourceApplyRMSPropOp<T>);

#define REGISTER_APPLY_RMSPROP(T)                           \
  REGISTER_KERNEL_BUILDER(Name("ApplyRMSProp")              \
                              .Device(DEVICE_MTGPU)         \
                              .TypeConstraint<T>("T")       \
                              .HostMemory("lr")             \
                              .HostMemory("rho")            \
                              .HostMemory("momentum")       \
                              .HostMemory("epsilon"),       \
                          MusaApplyRMSPropKernelOp<T>);

REGISTER_RESOURCE_RMSPROP(float);
REGISTER_RESOURCE_RMSPROP(double);
REGISTER_RESOURCE_RMSPROP(Eigen::half);
REGISTER_RESOURCE_RMSPROP(bfloat16);
REGISTER_RESOURCE_RMSPROP(int64);
REGISTER_RESOURCE_RMSPROP(int32);

REGISTER_APPLY_RMSPROP(float);
REGISTER_APPLY_RMSPROP(double);
REGISTER_APPLY_RMSPROP(Eigen::half);
REGISTER_APPLY_RMSPROP(bfloat16);

#undef REGISTER_RESOURCE_RMSPROP
#undef REGISTER_APPLY_RMSPROP

// CenteredRMSProp Op using resource variables
// CenteredRMSProp update formulas:
//   mg <- rho * mg_{t-1} + (1-rho) * grad
//   ms <- rho * ms_{t-1} + (1-rho) * grad^2
//   mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg^2 + epsilon)
//   var <- var - mom
template <typename T>
class MusaResourceApplyCenteredRMSPropOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyCenteredRMSPropOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> mg;
    core::RefCountPtr<Var> ms;
    core::RefCountPtr<Var> mom;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &mg));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &ms));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &mom));

    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(var->mu());
    add_mutex(mg->mu());
    add_mutex(ms->mu());
    add_mutex(mom->mu());
    std::sort(mutexes.begin(), mutexes.end());

    for (mutex* mu : mutexes) {
      mu->lock();
    }
    std::vector<MutexUnlockerRMSProp> locks;
    locks.reserve(mutexes.size());
    for (mutex* mu : mutexes) {
      locks.emplace_back(mu);
    }

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    mg->tensor()->IsInitialized() &&
                    ms->tensor()->IsInitialized() &&
                    mom->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "CenteredRMSProp variables (var/mg/ms/mom) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, mg.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, ms.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdateRMSProp(ctx, mom.get()));

    Tensor var_t = *var->tensor();
    Tensor mg_t = *mg->tensor();
    Tensor ms_t = *ms->tensor();
    Tensor mom_t = *mom->tensor();

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyCenteredRMSProp ", op_name,
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

    // Inputs: lr (4), rho (5), momentum (6), epsilon (7), grad (8)
    const T lr = ctx->input(4).scalar<T>()();
    const T rho = ctx->input(5).scalar<T>()();
    const T momentum = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();
    const Tensor& grad = ctx->input(8);

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_mg = CreateMTensor(mg_t, format_);
    mTensor t_ms = CreateMTensor(ms_t, format_);
    mTensor t_mom = CreateMTensor(mom_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    // Step 1: mg <- rho * mg + (1-rho) * grad
    mTensor t_rho;
    OP_REQUIRES_OK(ctx, fill_scalar(rho, mg_t.shape(), &t_rho));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mg, t_mg, t_rho),
                                        "MUL rho_mg"));

    mTensor t_one_minus_rho;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - rho, grad.shape(),
                                    &t_one_minus_rho));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_grad_scaled, t_grad,
                                            t_one_minus_rho),
                                   "MUL grad_scaled"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mg, t_mg, t_grad_scaled),
                                        "ADD mg"));

    // Step 2: ms <- rho * ms + (1-rho) * grad^2
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_ms, t_ms, t_rho),
                                        "MUL rho_ms"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_sq, t_grad, t_grad),
                                        "MUL grad_sq"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_sq_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_grad_sq_scaled, t_grad_sq,
                                            t_one_minus_rho),
                                   "MUL grad_sq_scaled"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_ms, t_ms, t_grad_sq_scaled),
                                        "ADD ms"));

    // Step 3: mom <- momentum * mom + lr * grad / sqrt(ms - mg^2 + epsilon)
    // First: momentum * mom
    mTensor t_momentum;
    OP_REQUIRES_OK(ctx, fill_scalar(momentum, mom_t.shape(), &t_momentum));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mom, t_mom, t_momentum),
                                        "MUL momentum_mom"));

    // Second: mg^2
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           mg_t.shape(), &temp_storage.back()));
    mTensor t_mg_sq = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mg_sq, t_mg, t_mg),
                                        "MUL mg_sq"));

    // Third: ms - mg^2
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           ms_t.shape(), &temp_storage.back()));
    mTensor t_ms_minus_mg_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_ms_minus_mg_sq, t_ms, t_mg_sq),
                                   "SUB ms_mg_sq"));

    // Fourth: ms - mg^2 + epsilon
    mTensor t_eps;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, ms_t.shape(), &t_eps));
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_ms_minus_mg_sq, t_ms_minus_mg_sq, t_eps),
                                   "ADD epsilon_centered"));

    // Fifth: sqrt(ms - mg^2 + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           ms_t.shape(), &temp_storage.back()));
    mTensor t_sqrt_denom = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(ctx, require_success(u_op.Run(handle, t_sqrt_denom, t_ms_minus_mg_sq),
                                        "SQRT centered_denom"));

    // Sixth: grad / sqrt(ms - mg^2 + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_div, t_grad, t_sqrt_denom),
                                        "DIV grad_centered"));

    // Seventh: lr * grad / sqrt(ms - mg^2 + epsilon)
    mTensor t_lr;
    OP_REQUIRES_OK(ctx, fill_scalar(lr, grad.shape(), &t_lr));
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_lr_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_lr_grad_div, t_grad_div, t_lr),
                                        "MUL lr_grad_centered"));

    // Eighth: mom <- momentum * mom + lr * grad / sqrt(ms - mg^2 + epsilon)
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_mom, t_mom, t_lr_grad_div),
                                        "ADD mom_centered"));

    // Step 4: var <- var - mom
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_var, t_var, t_mom),
                                        "SUB var_centered"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyCenteredRMSProp: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

// ApplyCenteredRMSProp Op using Ref tensors (non-resource version)
template <typename T>
class MusaApplyCenteredRMSPropKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyCenteredRMSPropKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    Var* var = nullptr;
    Var* mg = nullptr;
    Var* ms = nullptr;
    Var* mom = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &mg));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &ms));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 3), &mom));
    core::ScopedUnref var_unref(var), mg_unref(mg), ms_unref(ms), mom_unref(mom);

    Tensor* var_t = var->tensor();
    Tensor* mg_t = mg->tensor();
    Tensor* ms_t = ms->tensor();
    Tensor* mom_t = mom->tensor();

    OP_REQUIRES(ctx,
                var_t->IsInitialized() && mg_t->IsInitialized() &&
                    ms_t->IsInitialized() && mom_t->IsInitialized(),
                errors::FailedPrecondition(
                    "CenteredRMSProp variables (var/mg/ms/mom) not initialized."));

    // Inputs: lr (4), rho (5), momentum (6), epsilon (7), grad (8)
    const T lr = ctx->input(4).scalar<T>()();
    const T rho = ctx->input(5).scalar<T>()();
    const T momentum = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();
    const Tensor& grad = ctx->input(8);

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var = CreateMTensor(*var_t, format_);
    mTensor t_mg = CreateMTensor(*mg_t, format_);
    mTensor t_ms = CreateMTensor(*ms_t, format_);
    mTensor t_mom = CreateMTensor(*mom_t, format_);
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

    // Step 1: mg <- rho * mg + (1-rho) * grad
    mTensor t_rho;
    fill_scalar(rho, mg_t->shape(), &t_rho);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_mg, t_mg, t_rho);

    mTensor t_one_minus_rho;
    fill_scalar(static_cast<T>(1.0) - rho, grad.shape(), &t_one_minus_rho);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_scaled, t_grad, t_one_minus_rho);

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_mg, t_mg, t_grad_scaled);

    // Step 2: ms <- rho * ms + (1-rho) * grad^2
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_ms, t_ms, t_rho);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq, t_grad, t_grad);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_sq_scaled = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_grad_sq_scaled, t_grad_sq, t_one_minus_rho);

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_ms, t_ms, t_grad_sq_scaled);

    // Step 3: mom <- momentum * mom + lr * grad / sqrt(ms - mg^2 + epsilon)
    mTensor t_momentum;
    fill_scalar(momentum, mom_t->shape(), &t_momentum);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_mom, t_mom, t_momentum);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, mg_t->shape(),
                       &temp_storage.back());
    mTensor t_mg_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.Run(handle, t_mg_sq, t_mg, t_mg);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, ms_t->shape(),
                       &temp_storage.back());
    mTensor t_ms_minus_mg_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_ms_minus_mg_sq, t_ms, t_mg_sq);

    mTensor t_eps;
    fill_scalar(epsilon, ms_t->shape(), &t_eps);
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_ms_minus_mg_sq, t_ms_minus_mg_sq, t_eps);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, ms_t->shape(),
                       &temp_storage.back());
    mTensor t_sqrt_denom = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    u_op.Run(handle, t_sqrt_denom, t_ms_minus_mg_sq);

    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    b_op.Run(handle, t_grad_div, t_grad, t_sqrt_denom);

    mTensor t_lr;
    fill_scalar(lr, grad.shape(), &t_lr);
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad.shape(),
                       &temp_storage.back());
    mTensor t_lr_grad_div = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_lr_grad_div, t_grad_div, t_lr);

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    b_op.Run(handle, t_mom, t_mom, t_lr_grad_div);

    // Step 4: var <- var - mom
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_mom);

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
};

#define REGISTER_RESOURCE_CENTERED_RMSPROP(T)                        \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyCenteredRMSProp")       \
                              .Device(DEVICE_MTGPU)                  \
                              .HostMemory("var")                     \
                              .HostMemory("mg")                      \
                              .HostMemory("ms")                      \
                              .HostMemory("mom")                     \
                              .TypeConstraint<T>("T")                \
                              .HostMemory("lr")                      \
                              .HostMemory("rho")                     \
                              .HostMemory("momentum")                \
                              .HostMemory("epsilon"),                \
                          MusaResourceApplyCenteredRMSPropOp<T>);

#define REGISTER_APPLY_CENTERED_RMSPROP(T)                           \
  REGISTER_KERNEL_BUILDER(Name("ApplyCenteredRMSProp")               \
                              .Device(DEVICE_MTGPU)                  \
                              .TypeConstraint<T>("T")                \
                              .HostMemory("lr")                      \
                              .HostMemory("rho")                     \
                              .HostMemory("momentum")                \
                              .HostMemory("epsilon"),                \
                          MusaApplyCenteredRMSPropKernelOp<T>);

REGISTER_RESOURCE_CENTERED_RMSPROP(float);
REGISTER_RESOURCE_CENTERED_RMSPROP(double);
REGISTER_RESOURCE_CENTERED_RMSPROP(Eigen::half);
REGISTER_RESOURCE_CENTERED_RMSPROP(bfloat16);
REGISTER_RESOURCE_CENTERED_RMSPROP(int64);
REGISTER_RESOURCE_CENTERED_RMSPROP(int32);

REGISTER_APPLY_CENTERED_RMSPROP(float);
REGISTER_APPLY_CENTERED_RMSPROP(double);
REGISTER_APPLY_CENTERED_RMSPROP(Eigen::half);
REGISTER_APPLY_CENTERED_RMSPROP(bfloat16);

#undef REGISTER_RESOURCE_CENTERED_RMSPROP
#undef REGISTER_APPLY_CENTERED_RMSPROP

}  // namespace musa
}  // namespace tensorflow