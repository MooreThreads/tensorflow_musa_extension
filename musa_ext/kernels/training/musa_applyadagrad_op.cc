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

// MUSA kernel for Adagrad update:
// accum = accum + grad * grad
// var = var - lr * grad / (sqrt(accum) + epsilon)
template <typename T>
class MusaResourceApplyAdagradV2Op : public MusaOpKernel {
 public:
  explicit MusaResourceApplyAdagradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> accum;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));

    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(var->mu());
    add_mutex(accum->mu());
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
                var->tensor()->IsInitialized() && accum->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Adagrad variables (var/accum) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, accum.get()));

    Tensor var_t = *var->tensor();
    Tensor accum_t = *accum->tensor();

    // Get learning rate (scalar on host memory)
    const Tensor& lr_t = ctx->input(2);
    OP_REQUIRES(ctx, lr_t.NumElements() == 1,
                errors::InvalidArgument("lr must be a scalar, got ",
                                        lr_t.NumElements(), " elements"));

    // Get epsilon (scalar on host memory)
    const Tensor& epsilon_t = ctx->input(3);
    OP_REQUIRES(ctx, epsilon_t.NumElements() == 1,
                errors::InvalidArgument("epsilon must be a scalar, got ",
                                        epsilon_t.NumElements(), " elements"));

    // Get gradient
    const Tensor& grad_t = ctx->input(4);

    OP_REQUIRES(ctx, var_t.shape().IsSameSize(grad_t.shape()),
                errors::InvalidArgument(
                    "Variable and gradient must have the same shape. var: ",
                    var_t.shape().DebugString(), " grad: ",
                    grad_t.shape().DebugString()));

    OP_REQUIRES(ctx, accum_t.shape().IsSameSize(var_t.shape()),
                errors::InvalidArgument(
                    "Variable and accum must have the same shape. var: ",
                    var_t.shape().DebugString(), " accum: ",
                    accum_t.shape().DebugString()));

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_accum = CreateMTensor(accum_t, format_);
    mTensor t_grad = CreateMTensor(grad_t, format_);

    const T lr = lr_t.scalar<T>()();
    const T epsilon = epsilon_t.scalar<T>()();

    // Helper lambda to convert MuDNN Status to TensorFlow Status
    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyAdagradV2 ", op_name,
                                " failed. Status: ", static_cast<int>(status));
      }
      return Status::OK();
    };

    auto fill_scalar = [&](T val, const TensorShape& shape, mTensor* out) -> Status {
      temp_storage.emplace_back();
      Status alloc_status = ctx->allocate_temp(DataTypeToEnum<T>::value, shape,
                                               &temp_storage.back());
      if (!alloc_status.ok()) {
        return alloc_status;
      }
      *out = CreateMTensor(temp_storage.back(), format_);
      return MusaFillCall(out, val, ctx);
    };

    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    // Step 1: Compute grad_sq = grad * grad
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad_t.shape(), &temp_storage.back()));
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_sq, t_grad, t_grad),
                                        "MUL grad_sq"));

    // Step 2: accum = accum + grad_sq
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_accum, t_accum, t_grad_sq),
                                        "ADD accum"));

    // Step 3: sqrt_accum = sqrt(accum)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           accum_t.shape(), &temp_storage.back()));
    mTensor t_sqrt_accum = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(ctx, require_success(u_op.Run(handle, t_sqrt_accum, t_accum),
                                        "SQRT accum"));

    // Step 4: denominator = sqrt_accum + epsilon
    mTensor t_epsilon;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, accum_t.shape(), &t_epsilon));
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_sqrt_accum, t_sqrt_accum, t_epsilon),
                                        "ADD epsilon"));

    // Step 5: update = grad / denominator
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad_t.shape(), &temp_storage.back()));
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_update, t_grad, t_sqrt_accum),
                                        "DIV update"));

    // Step 6: scaled_update = lr * update
    mTensor t_lr;
    OP_REQUIRES_OK(ctx, fill_scalar(lr, grad_t.shape(), &t_lr));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_update, t_update, t_lr),
                                        "MUL lr"));

    // Step 7: var = var - scaled_update
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_var, t_var, t_update),
                                        "SUB var"));

    // Synchronize to ensure computation is complete
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyAdagradV2: "
                                 "musaStreamSynchronize failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

// ApplyAdagradV2 (non-resource version)
template <typename T>
class MusaApplyAdagradV2Op : public MusaOpKernel {
 public:
  explicit MusaApplyAdagradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& var_t = ctx->input(0);
    const Tensor& accum_t = ctx->input(1);
    const Tensor& lr_t = ctx->input(2);
    const Tensor& epsilon_t = ctx->input(3);
    const Tensor& grad_t = ctx->input(4);

    OP_REQUIRES(ctx, lr_t.NumElements() == 1,
                errors::InvalidArgument("lr must be a scalar"));

    OP_REQUIRES(ctx, epsilon_t.NumElements() == 1,
                errors::InvalidArgument("epsilon must be a scalar"));

    OP_REQUIRES(ctx, var_t.shape().IsSameSize(grad_t.shape()),
                errors::InvalidArgument(
                    "var and grad must have the same shape"));

    OP_REQUIRES(ctx, accum_t.shape().IsSameSize(var_t.shape()),
                errors::InvalidArgument(
                    "var and accum must have the same shape"));

    // Allocate outputs
    Tensor* var_out;
    Tensor* accum_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, var_t.shape(), &var_out));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, accum_t.shape(), &accum_out));

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var_in = CreateMTensor(var_t, format_);
    mTensor t_accum_in = CreateMTensor(accum_t, format_);
    mTensor t_grad = CreateMTensor(grad_t, format_);
    mTensor t_var_out = CreateMTensor(*var_out, format_);
    mTensor t_accum_out = CreateMTensor(*accum_out, format_);

    const T lr = lr_t.scalar<T>()();
    const T epsilon = epsilon_t.scalar<T>()();

    // Helper lambda to convert MuDNN Status to TensorFlow Status
    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ApplyAdagradV2 ", op_name,
                                " failed. Status: ", static_cast<int>(status));
      }
      return Status::OK();
    };

    auto fill_scalar = [&](T val, const TensorShape& shape, mTensor* out) -> Status {
      temp_storage.emplace_back();
      Status alloc_status = ctx->allocate_temp(DataTypeToEnum<T>::value, shape,
                                               &temp_storage.back());
      if (!alloc_status.ok()) {
        return alloc_status;
      }
      *out = CreateMTensor(temp_storage.back(), format_);
      return MusaFillCall(out, val, ctx);
    };

    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    // Copy inputs to outputs first (using ADD with zero)
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    mTensor t_zero;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(0.0), var_t.shape(), &t_zero));
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_var_out, t_var_in, t_zero),
                                        "COPY var"));
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_accum_out, t_accum_in, t_zero),
                                        "COPY accum"));

    // Step 1: Compute grad_sq = grad * grad
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad_t.shape(), &temp_storage.back()));
    mTensor t_grad_sq = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_grad_sq, t_grad, t_grad),
                                        "MUL grad_sq"));

    // Step 2: accum_out = accum_out + grad_sq
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_accum_out, t_accum_out, t_grad_sq),
                                        "ADD accum"));

    // Step 3: sqrt_accum = sqrt(accum_out)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           accum_t.shape(), &temp_storage.back()));
    mTensor t_sqrt_accum = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(ctx, require_success(u_op.Run(handle, t_sqrt_accum, t_accum_out),
                                        "SQRT accum"));

    // Step 4: denominator = sqrt_accum + epsilon
    mTensor t_epsilon;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, accum_t.shape(), &t_epsilon));
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_sqrt_accum, t_sqrt_accum, t_epsilon),
                                        "ADD epsilon"));

    // Step 5: update = grad / denominator
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad_t.shape(), &temp_storage.back()));
    mTensor t_update = CreateMTensor(temp_storage.back(), format_);
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_update, t_grad, t_sqrt_accum),
                                        "DIV update"));

    // Step 6: scaled_update = lr * update
    mTensor t_lr;
    OP_REQUIRES_OK(ctx, fill_scalar(lr, grad_t.shape(), &t_lr));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_update, t_update, t_lr),
                                        "MUL lr"));

    // Step 7: var_out = var_out - scaled_update
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_var_out, t_var_out, t_update),
                                        "SUB var"));

    // Forward input to output for reference types
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
      ctx->forward_ref_input_to_ref_output(1, 1);
    }
  }

 private:
  bool use_exclusive_lock_;
};

// Register the kernels for supported types
#define REGISTER_RESOURCE_ADAGRAD_V2(T)                        \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradV2")       \
                              .Device(DEVICE_MTGPU)            \
                              .HostMemory("lr")                \
                              .HostMemory("epsilon")           \
                              .TypeConstraint<T>("T"),         \
                          MusaResourceApplyAdagradV2Op<T>);

REGISTER_RESOURCE_ADAGRAD_V2(float);
REGISTER_RESOURCE_ADAGRAD_V2(Eigen::half);
REGISTER_RESOURCE_ADAGRAD_V2(bfloat16);
REGISTER_RESOURCE_ADAGRAD_V2(double);

#define REGISTER_APPLY_ADAGRAD_V2(T)                           \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdagradV2")               \
                              .Device(DEVICE_MTGPU)            \
                              .HostMemory("lr")                \
                              .HostMemory("epsilon")           \
                              .TypeConstraint<T>("T"),         \
                          MusaApplyAdagradV2Op<T>);

REGISTER_APPLY_ADAGRAD_V2(float);
REGISTER_APPLY_ADAGRAD_V2(Eigen::half);
REGISTER_APPLY_ADAGRAD_V2(bfloat16);
REGISTER_APPLY_ADAGRAD_V2(double);

#undef REGISTER_RESOURCE_ADAGRAD_V2
#undef REGISTER_APPLY_ADAGRAD_V2

}  // namespace musa
}  // namespace tensorflow