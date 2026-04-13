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

// MUSA kernel for SGD update: var = var - lr * grad
// Uses MuDNN operations for computation
template <typename T>
class MusaResourceApplyGradientDescentOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyGradientDescentOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

    mutex* mu = var->mu();
    mu->lock();
    MutexUnlocker unlocker(mu);

    OP_REQUIRES(ctx, var->tensor()->IsInitialized(),
                errors::FailedPrecondition("Variable not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));

    Tensor var_t = *var->tensor();

    // Get learning rate (scalar on host memory)
    const Tensor& lr_t = ctx->input(1);
    OP_REQUIRES(ctx, lr_t.NumElements() == 1,
                errors::InvalidArgument("lr must be a scalar, got ",
                                        lr_t.NumElements(), " elements"));

    // Get gradient
    const Tensor& grad_t = ctx->input(2);

    OP_REQUIRES(ctx, var_t.shape().IsSameSize(grad_t.shape()),
                errors::InvalidArgument(
                    "Variable and gradient must have the same shape. var: ",
                    var_t.shape().DebugString(), " grad: ",
                    grad_t.shape().DebugString()));

    // Get MuDNN handle
    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    // Create mTensor wrappers
    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_grad = CreateMTensor(grad_t, format_);

    // Get scalar learning rate value
    const T lr = lr_t.scalar<T>()();

    // Fill lr scalar tensor with proper precision preservation
    // Use double for fill value to preserve precision for all types
    auto fill_scalar = [&](double val, const TensorShape& shape, mTensor* out) {
      temp_storage.emplace_back();
      ctx->allocate_temp(DataTypeToEnum<T>::value, shape, &temp_storage.back());
      *out = CreateMTensor(temp_storage.back(), format_);
      ::musa::dnn::Fill fill_op;
      // SetValue accepts double, which preserves precision for float64
      fill_op.SetValue(val);
      return fill_op.Run(handle, *out);
    };

    // Create lr tensor for multiplication
    mTensor t_lr;
    fill_scalar(static_cast<double>(lr), grad_t.shape(), &t_lr);

    // Compute: lr * grad
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad_t.shape(),
                       &temp_storage.back());
    mTensor t_lr_grad = CreateMTensor(temp_storage.back(), format_);

    ::musa::dnn::Binary b_op;
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_lr_grad, t_grad, t_lr);

    // Compute: var = var - lr * grad
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_var, t_var, t_lr_grad);

    // Synchronize to ensure computation is complete
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyGradientDescent: "
                                 "musaStreamSynchronize failed: ",
                                 musaGetErrorString(sync_err)));
  }

 private:
  bool use_exclusive_lock_;
};

// Register the kernel for supported types
#define REGISTER_RESOURCE_GRADIENT_DESCENT(T)                     \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyGradientDescent")    \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("lr")                   \
                              .TypeConstraint<T>("T"),            \
                          MusaResourceApplyGradientDescentOp<T>);

REGISTER_RESOURCE_GRADIENT_DESCENT(float);
REGISTER_RESOURCE_GRADIENT_DESCENT(Eigen::half);
REGISTER_RESOURCE_GRADIENT_DESCENT(bfloat16);

// Note: muDNN does not support DOUBLE (float64) for binary operations (MUL, SUB).
// Do not register for double - TensorFlow will fall back to CPU implementation.

// ApplyGradientDescent (non-resource version)
template <typename T>
class MusaApplyGradientDescentOp : public MusaOpKernel {
 public:
  explicit MusaApplyGradientDescentOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& var_t = ctx->input(0);
    const Tensor& lr_t = ctx->input(1);
    const Tensor& grad_t = ctx->input(2);

    OP_REQUIRES(ctx, lr_t.NumElements() == 1,
                errors::InvalidArgument("lr must be a scalar"));

    OP_REQUIRES(ctx, var_t.shape().IsSameSize(grad_t.shape()),
                errors::InvalidArgument(
                    "var and grad must have the same shape"));

    // Allocate output
    Tensor* out_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, var_t.shape(), &out_t));

    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_grad = CreateMTensor(grad_t, format_);
    mTensor t_out = CreateMTensor(*out_t, format_);

    const T lr = lr_t.scalar<T>()();

    // Fill lr scalar tensor with proper precision preservation
    // Use double for fill value to preserve precision for all types
    auto fill_scalar = [&](double val, const TensorShape& shape, mTensor* out) {
      temp_storage.emplace_back();
      ctx->allocate_temp(DataTypeToEnum<T>::value, shape, &temp_storage.back());
      *out = CreateMTensor(temp_storage.back(), format_);
      ::musa::dnn::Fill fill_op;
      // SetValue accepts double, which preserves precision for float64
      fill_op.SetValue(val);
      return fill_op.Run(handle, *out);
    };

    mTensor t_lr;
    fill_scalar(static_cast<double>(lr), grad_t.shape(), &t_lr);

    // Compute: lr * grad
    temp_storage.emplace_back();
    ctx->allocate_temp(DataTypeToEnum<T>::value, grad_t.shape(),
                       &temp_storage.back());
    mTensor t_lr_grad = CreateMTensor(temp_storage.back(), format_);

    ::musa::dnn::Binary b_op;
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    b_op.Run(handle, t_lr_grad, t_grad, t_lr);

    // Compute: out = var - lr * grad
    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    b_op.Run(handle, t_out, t_var, t_lr_grad);

    // Forward input to output for reference types
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }
};

#define REGISTER_APPLY_GRADIENT_DESCENT(T)                        \
  REGISTER_KERNEL_BUILDER(Name("ApplyGradientDescent")            \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<T>("T"),            \
                          MusaApplyGradientDescentOp<T>);

REGISTER_APPLY_GRADIENT_DESCENT(float);
REGISTER_APPLY_GRADIENT_DESCENT(Eigen::half);
REGISTER_APPLY_GRADIENT_DESCENT(bfloat16);
// Note: muDNN does not support DOUBLE for binary operations. Not registered.

}  // namespace musa
}  // namespace tensorflow