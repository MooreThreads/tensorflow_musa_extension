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
#include <stdint.h>
#include <limits>
#include <list>
#include <vector>

#include "../array/musa_fill_functor.h"
#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

// Helper function declarations (defined in musa_applyadam_op.cc)
extern Status CopyTensorForUpdate(OpKernelContext* ctx, const Tensor& src,
                                  Tensor* dst);

extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

extern "C" void LaunchResourceApplyGradientDescentFloat(
    float* var, const float* grad, float alpha, int64_t n,
    musaStream_t stream);

extern "C" void LaunchResourceApplyGradientDescentFloatBFloat16(
    float* var, const void* grad, float alpha, int64_t n,
    musaStream_t stream);

extern "C" void LaunchGroupedResourceApplyGradientDescentFloatBFloat16(
    const uint64_t* var_ptrs, const uint64_t* grad_ptrs,
    const float* alphas, const int64_t* sizes, const int64_t* block_offsets,
    int num_tensors, int total_blocks, musaStream_t stream);

extern "C" void LaunchGroupedResourceApplyGradientDescentFloat(
    const uint64_t* var_ptrs, const uint64_t* grad_ptrs,
    const float* alphas, const int64_t* sizes, const int64_t* block_offsets,
    int num_tensors, int total_blocks, musaStream_t stream);

constexpr int kGroupedApplyThreads = 256;
constexpr int kGroupedApplyMaxBlocksPerTensor = 4096;

int GroupedApplyBlocksForNumElements(int64_t n) {
  int blocks = static_cast<int>((n + kGroupedApplyThreads - 1) /
                                kGroupedApplyThreads);
  if (blocks < 1) return 1;
  return blocks > kGroupedApplyMaxBlocksPerTensor
             ? kGroupedApplyMaxBlocksPerTensor
             : blocks;
}

template <typename T>
bool TryLaunchFastResourceApplyGradientDescent(OpKernelContext* ctx,
                                               Tensor* var_t,
                                               const Tensor& lr_t,
                                               const Tensor& grad_t) {
  return false;
}

template <>
bool TryLaunchFastResourceApplyGradientDescent<float>(
    OpKernelContext* ctx, Tensor* var_t, const Tensor& lr_t,
    const Tensor& grad_t) {
  LaunchResourceApplyGradientDescentFloat(
      var_t->flat<float>().data(), grad_t.flat<float>().data(),
      lr_t.scalar<float>()(), static_cast<int64_t>(var_t->NumElements()),
      GetMusaStreamByCtx(ctx));
  return true;
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

class MutexListUnlocker {
 public:
  explicit MutexListUnlocker(std::vector<mutex*>* locks) : locks_(locks) {}
  ~MutexListUnlocker() {
    if (locks_ == nullptr) return;
    for (auto it = locks_->rbegin(); it != locks_->rend(); ++it) {
      (*it)->unlock();
    }
  }

 private:
  std::vector<mutex*>* locks_;
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
                    var_t.shape().DebugString(),
                    " grad: ", grad_t.shape().DebugString()));

    if (TryLaunchFastResourceApplyGradientDescent<T>(ctx, &var_t, lr_t,
                                                     grad_t)) {
      return;
    }

    // Get MuDNN handle
    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
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
  }

 private:
  bool use_exclusive_lock_;
};

// Register the kernel for supported types
#define REGISTER_RESOURCE_GRADIENT_DESCENT(T)                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyGradientDescent") \
                              .Device(DEVICE_MTGPU)            \
                              .HostMemory("alpha")             \
                              .TypeConstraint<T>("T"),         \
                          MusaResourceApplyGradientDescentOp<T>);

REGISTER_RESOURCE_GRADIENT_DESCENT(float);
REGISTER_RESOURCE_GRADIENT_DESCENT(Eigen::half);
REGISTER_RESOURCE_GRADIENT_DESCENT(bfloat16);

// Note: muDNN does not support DOUBLE (float64) for binary operations (MUL,
// SUB). Do not register for double - TensorFlow will fall back to CPU
// implementation.

REGISTER_OP("MusaResourceApplyGradientDescentMixed")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("delta: Tgrad")
    .Attr("T: {float}")
    .Attr("Tgrad: {bfloat16}")
    .Attr("use_locking: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

class MusaResourceApplyGradientDescentMixedOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyGradientDescentMixedOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

    mutex* mu = var->mu();
    mu->lock();
    MutexUnlocker unlocker(mu);

    OP_REQUIRES(ctx, var->tensor()->IsInitialized(),
                errors::FailedPrecondition("Variable not initialized."));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));

    Tensor var_t = *var->tensor();
    const Tensor& alpha_t = ctx->input(1);
    const Tensor& grad_t = ctx->input(2);

    OP_REQUIRES(ctx, alpha_t.NumElements() == 1,
                errors::InvalidArgument("alpha must be a scalar, got ",
                                        alpha_t.NumElements(), " elements"));
    OP_REQUIRES(ctx, var_t.dtype() == DT_FLOAT,
                errors::InvalidArgument(
                    "MusaResourceApplyGradientDescentMixed expects float var"));
    OP_REQUIRES(ctx, grad_t.dtype() == DT_BFLOAT16,
                errors::InvalidArgument(
                    "MusaResourceApplyGradientDescentMixed expects bfloat16 "
                    "gradient"));
    OP_REQUIRES(ctx, var_t.shape().IsSameSize(grad_t.shape()),
                errors::InvalidArgument(
                    "Variable and gradient must have the same shape. var: ",
                    var_t.shape().DebugString(),
                    " grad: ", grad_t.shape().DebugString()));

    const int64_t n = var_t.NumElements();
    if (n == 0) return;

    const float alpha = alpha_t.scalar<float>()();
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchResourceApplyGradientDescentFloatBFloat16(
        var_t.flat<float>().data(),
        reinterpret_cast<const void*>(grad_t.flat<bfloat16>().data()), alpha,
        n, stream);
  }

 private:
  bool use_exclusive_lock_;
};

REGISTER_KERNEL_BUILDER(
    Name("MusaResourceApplyGradientDescentMixed")
        .Device(DEVICE_MTGPU)
        .HostMemory("var")
        .HostMemory("alpha")
        .TypeConstraint<float>("T")
        .TypeConstraint<bfloat16>("Tgrad"),
    MusaResourceApplyGradientDescentMixedOp);

REGISTER_OP("MusaGroupedResourceApplyGradientDescentMixed")
    .Input("vars: N * resource")
    .Input("alphas: N * T")
    .Input("deltas: N * Tgrad")
    .Attr("N: int >= 1")
    .Attr("T: {float}")
    .Attr("Tgrad: {float, bfloat16}")
    .Attr("use_locking: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

class MusaGroupedResourceApplyGradientDescentMixedOp : public MusaOpKernel {
 public:
  explicit MusaGroupedResourceApplyGradientDescentMixedOp(
      OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<core::RefCountPtr<Var>> vars(n_);
    std::vector<mutex*> locks;
    locks.reserve(n_);
    MutexListUnlocker unlocker(&locks);
    std::vector<uint64_t> var_ptrs(n_);
    std::vector<uint64_t> grad_ptrs(n_);
    std::vector<float> alphas(n_);
    std::vector<int64_t> sizes(n_);
    std::vector<int64_t> block_offsets(n_ + 1, 0);

    for (int i = 0; i < n_; ++i) {
      const Tensor& alpha_t = ctx->input(n_ + i);
      OP_REQUIRES(ctx, alpha_t.NumElements() == 1,
                  errors::InvalidArgument("alpha must be a scalar, got ",
                                          alpha_t.NumElements(), " elements"));
      alphas[i] = alpha_t.scalar<float>()();

      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, i),
                                         &vars[i]));
      mutex* mu = vars[i]->mu();
      mu->lock();
      locks.push_back(mu);
      OP_REQUIRES(ctx, vars[i]->tensor()->IsInitialized(),
                  errors::FailedPrecondition("Variable not initialized."));
      OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, vars[i].get()));

      Tensor* var_t = vars[i]->tensor();
      const Tensor& grad_t = ctx->input(2 * n_ + i);
      OP_REQUIRES(ctx, var_t->dtype() == DT_FLOAT,
                  errors::InvalidArgument(
                      "MusaGroupedResourceApplyGradientDescentMixed expects "
                      "float variables"));
      OP_REQUIRES(ctx,
                  grad_t.dtype() == DT_FLOAT || grad_t.dtype() == DT_BFLOAT16,
                  errors::InvalidArgument(
                      "MusaGroupedResourceApplyGradientDescentMixed expects "
                      "float or bfloat16 gradients"));
      OP_REQUIRES(ctx, var_t->shape().IsSameSize(grad_t.shape()),
                  errors::InvalidArgument(
                      "Variable and gradient must have the same shape. var: ",
                      var_t->shape().DebugString(),
                      " grad: ", grad_t.shape().DebugString()));

      sizes[i] = var_t->NumElements();
      var_ptrs[i] =
          reinterpret_cast<uint64_t>(var_t->flat<float>().data());
      if (grad_t.dtype() == DT_BFLOAT16) {
        grad_ptrs[i] =
            reinterpret_cast<uint64_t>(grad_t.flat<bfloat16>().data());
      } else {
        grad_ptrs[i] =
            reinterpret_cast<uint64_t>(grad_t.flat<float>().data());
      }
      block_offsets[i + 1] =
          block_offsets[i] + GroupedApplyBlocksForNumElements(sizes[i]);
    }

    const int64_t total_blocks64 = block_offsets[n_];
    if (total_blocks64 <= 0) return;
    OP_REQUIRES(ctx, total_blocks64 <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    "Grouped apply total block count is too large: ",
                    total_blocks64));

    Tensor d_var_ptrs;
    Tensor d_grad_ptrs;
    Tensor d_alphas;
    Tensor d_sizes;
    Tensor d_block_offsets;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_UINT64, TensorShape({n_}),
                                      &d_var_ptrs));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_UINT64, TensorShape({n_}),
                                      &d_grad_ptrs));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, TensorShape({n_}),
                                      &d_alphas));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT64, TensorShape({n_}), &d_sizes));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT64, TensorShape({n_ + 1}),
                                      &d_block_offsets));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    auto copy_h2d = [&](void* dst, const void* src, size_t bytes,
                        const char* name) {
      mStatus status = MusaMemcpyAsyncH2D(dst, src, bytes, stream);
      OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                  errors::Internal(name, " metadata copy failed"));
    };
    copy_h2d(d_var_ptrs.flat<uint64>().data(), var_ptrs.data(),
             var_ptrs.size() * sizeof(uint64_t), "var_ptrs");
    copy_h2d(d_grad_ptrs.flat<uint64>().data(), grad_ptrs.data(),
             grad_ptrs.size() * sizeof(uint64_t), "grad_ptrs");
    copy_h2d(d_alphas.flat<float>().data(), alphas.data(),
             alphas.size() * sizeof(float), "alphas");
    copy_h2d(d_sizes.flat<int64>().data(), sizes.data(),
             sizes.size() * sizeof(int64_t), "sizes");
    copy_h2d(d_block_offsets.flat<int64>().data(), block_offsets.data(),
             block_offsets.size() * sizeof(int64_t), "block_offsets");

    if (ctx->input(2 * n_).dtype() == DT_BFLOAT16) {
      LaunchGroupedResourceApplyGradientDescentFloatBFloat16(
          reinterpret_cast<const uint64_t*>(d_var_ptrs.flat<uint64>().data()),
          reinterpret_cast<const uint64_t*>(d_grad_ptrs.flat<uint64>().data()),
          d_alphas.flat<float>().data(), d_sizes.flat<int64>().data(),
          d_block_offsets.flat<int64>().data(), n_,
          static_cast<int>(total_blocks64), stream);
    } else {
      LaunchGroupedResourceApplyGradientDescentFloat(
          reinterpret_cast<const uint64_t*>(d_var_ptrs.flat<uint64>().data()),
          reinterpret_cast<const uint64_t*>(d_grad_ptrs.flat<uint64>().data()),
          d_alphas.flat<float>().data(), d_sizes.flat<int64>().data(),
          d_block_offsets.flat<int64>().data(), n_,
          static_cast<int>(total_blocks64), stream);
    }
  }

 private:
  int n_ = 0;
  bool use_exclusive_lock_ = false;
};

REGISTER_KERNEL_BUILDER(
    Name("MusaGroupedResourceApplyGradientDescentMixed")
        .Device(DEVICE_MTGPU)
        .HostMemory("vars")
        .HostMemory("alphas")
        .TypeConstraint<float>("T")
        .TypeConstraint<float>("Tgrad"),
    MusaGroupedResourceApplyGradientDescentMixedOp);

REGISTER_KERNEL_BUILDER(
    Name("MusaGroupedResourceApplyGradientDescentMixed")
        .Device(DEVICE_MTGPU)
        .HostMemory("vars")
        .HostMemory("alphas")
        .TypeConstraint<float>("T")
        .TypeConstraint<bfloat16>("Tgrad"),
    MusaGroupedResourceApplyGradientDescentMixedOp);

// ApplyGradientDescent (non-resource version)
template <typename T>
class MusaApplyGradientDescentOp : public MusaOpKernel {
 public:
  explicit MusaApplyGradientDescentOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& var_t = ctx->input(0);
    const Tensor& lr_t = ctx->input(1);
    const Tensor& grad_t = ctx->input(2);

    OP_REQUIRES(ctx, lr_t.NumElements() == 1,
                errors::InvalidArgument("lr must be a scalar"));

    OP_REQUIRES(
        ctx, var_t.shape().IsSameSize(grad_t.shape()),
        errors::InvalidArgument("var and grad must have the same shape"));

    // Allocate output
    Tensor* out_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, var_t.shape(), &out_t));

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
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

#define REGISTER_APPLY_GRADIENT_DESCENT(T)             \
  REGISTER_KERNEL_BUILDER(Name("ApplyGradientDescent") \
                              .Device(DEVICE_MTGPU)    \
                              .TypeConstraint<T>("T"), \
                          MusaApplyGradientDescentOp<T>);

REGISTER_APPLY_GRADIENT_DESCENT(float);
REGISTER_APPLY_GRADIENT_DESCENT(Eigen::half);
REGISTER_APPLY_GRADIENT_DESCENT(bfloat16);
// Note: muDNN does not support DOUBLE for binary operations. Not registered.

}  // namespace musa
}  // namespace tensorflow
