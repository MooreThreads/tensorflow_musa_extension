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
#include <cstdint>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchFusedResourceApplyRMSPropKernel(
    T* var, T* ms, T* mom, const T* grad, double lr, double rho,
    double momentum, double epsilon, int64_t n, musaStream_t stream);

template <typename T>
void LaunchFusedResourceApplyCenteredRMSPropKernel(
    T* var, T* mg, T* ms, T* mom, const T* grad, double lr, double rho,
    double momentum, double epsilon, int64_t n, musaStream_t stream);

namespace {

template <typename T>
double ScalarToDouble(const Tensor& tensor) {
  return static_cast<double>(tensor.scalar<T>()());
}

template <>
double ScalarToDouble<Eigen::half>(const Tensor& tensor) {
  return static_cast<double>(static_cast<float>(tensor.scalar<Eigen::half>()()));
}

template <>
double ScalarToDouble<bfloat16>(const Tensor& tensor) {
  return static_cast<double>(static_cast<float>(tensor.scalar<bfloat16>()()));
}

Status CopyTensorForFusedRMSPropUpdate(OpKernelContext* ctx, const Tensor& src,
                                       Tensor* dst) {
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  attr.set_nic_compatible(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst, attr));

  if (src.TotalBytes() == 0) {
    return OkStatus();
  }

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(dst->data(), src.data(), src.TotalBytes(),
                                    musaMemcpyDeviceToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal(
        "CopyTensorForUpdateRMSProp: musaMemcpyAsync failed: ",
        musaGetErrorString(err));
  }

  return OkStatus();
}

Status PrepareTensorForFusedRMSPropUpdate(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return OkStatus();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(
      CopyTensorForFusedRMSPropUpdate(ctx, *var->tensor(), &copied));
  *var->tensor() = copied;
  return OkStatus();
}

class ScopedMutexUnlocker {
 public:
  explicit ScopedMutexUnlocker(mutex* mu) : mu_(mu) {}
  ScopedMutexUnlocker(ScopedMutexUnlocker&& other) noexcept : mu_(other.mu_) {
    other.mu_ = nullptr;
  }
  ScopedMutexUnlocker(const ScopedMutexUnlocker&) = delete;
  ScopedMutexUnlocker& operator=(const ScopedMutexUnlocker&) = delete;
  ~ScopedMutexUnlocker() {
    if (mu_ != nullptr) mu_->unlock();
  }

 private:
  mutex* mu_;
};

void LockResourceVariables(const std::vector<Var*>& vars,
                           std::vector<ScopedMutexUnlocker>* locks) {
  std::vector<mutex*> mutexes;
  mutexes.reserve(vars.size());
  for (Var* var : vars) {
    mutex* mu = var->mu();
    if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
      mutexes.push_back(mu);
    }
  }
  std::sort(mutexes.begin(), mutexes.end());

  locks->reserve(mutexes.size());
  for (mutex* mu : mutexes) {
    mu->lock();
    locks->emplace_back(mu);
  }
}

void LockRefInputs(OpKernelContext* ctx, const std::vector<int>& input_indices,
                   std::vector<ScopedMutexUnlocker>* locks) {
  std::vector<mutex*> mutexes;
  mutexes.reserve(input_indices.size());
  for (int index : input_indices) {
    mutex* mu = ctx->input_ref_mutex(index);
    if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
      mutexes.push_back(mu);
    }
  }
  std::sort(mutexes.begin(), mutexes.end());

  locks->reserve(mutexes.size());
  for (mutex* mu : mutexes) {
    mu->lock();
    locks->emplace_back(mu);
  }
}

Status CheckSameShape(const Tensor& lhs, const Tensor& rhs, const char* lhs_name,
                      const char* rhs_name) {
  if (!lhs.shape().IsSameSize(rhs.shape())) {
    return errors::InvalidArgument(lhs_name, " and ", rhs_name,
                                   " must have the same shape. ", lhs_name,
                                   " shape: ", lhs.shape().DebugString(), ", ",
                                   rhs_name,
                                   " shape: ", rhs.shape().DebugString());
  }
  return OkStatus();
}

Status CheckScalar(const Tensor& tensor, const char* name) {
  if (tensor.NumElements() != 1) {
    return errors::InvalidArgument(name, " must be a scalar, got ",
                                   tensor.NumElements(), " elements.");
  }
  return OkStatus();
}

void CheckLaunchAndSync(OpKernelContext* ctx, const char* op_name) {
  musaError_t launch_err = musaGetLastError();
  OP_REQUIRES(ctx, launch_err == musaSuccess,
              errors::Internal(op_name, ": MUSA kernel launch failed: ",
                               musaGetErrorString(launch_err)));

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t sync_err = musaStreamSynchronize(stream);
  OP_REQUIRES(ctx, sync_err == musaSuccess,
              errors::Internal(op_name, ": musaStreamSynchronize failed: ",
                               musaGetErrorString(sync_err)));
}

}  // namespace

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

    std::vector<ScopedMutexUnlocker> locks;
    LockResourceVariables({var.get(), ms.get(), mom.get()}, &locks);

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    ms->tensor()->IsInitialized() &&
                    mom->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Fused RMSProp variables (var/ms/mom) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, ms.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, mom.get()));

    Tensor* var_t = var->tensor();
    Tensor* ms_t = ms->tensor();
    Tensor* mom_t = mom->tensor();
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *ms_t, "var", "ms"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *mom_t, "var", "mom"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(3), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "rho"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "momentum"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "epsilon"));

    const int64_t n = var_t->NumElements();
    if (n == 0) return;

    const double lr = ScalarToDouble<T>(ctx->input(3));
    const double rho = ScalarToDouble<T>(ctx->input(4));
    const double momentum = ScalarToDouble<T>(ctx->input(5));
    const double epsilon = ScalarToDouble<T>(ctx->input(6));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchFusedResourceApplyRMSPropKernel<T>(
        var_t->flat<T>().data(), ms_t->flat<T>().data(),
        mom_t->flat<T>().data(), grad.flat<T>().data(), lr, rho, momentum,
        epsilon, n, stream);

    CheckLaunchAndSync(ctx, "ResourceApplyRMSProp");
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class MusaApplyRMSPropKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyRMSPropKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<ScopedMutexUnlocker> locks;
    if (use_exclusive_lock_) {
      LockRefInputs(ctx, {0, 1, 2}, &locks);
    }

    Tensor var_t = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms_t = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom_t = ctx->mutable_input(2, use_exclusive_lock_);
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES(ctx,
                var_t.IsInitialized() && ms_t.IsInitialized() &&
                    mom_t.IsInitialized(),
                errors::FailedPrecondition(
                    "Fused ApplyRMSProp variables (var/ms/mom) not initialized."));

    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, ms_t, "var", "ms"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, mom_t, "var", "mom"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(3), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "rho"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "momentum"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "epsilon"));

    const int64_t n = var_t.NumElements();
    if (n != 0) {
      const double lr = ScalarToDouble<T>(ctx->input(3));
      const double rho = ScalarToDouble<T>(ctx->input(4));
      const double momentum = ScalarToDouble<T>(ctx->input(5));
      const double epsilon = ScalarToDouble<T>(ctx->input(6));

      musaStream_t stream = GetMusaStreamByCtx(ctx);
      LaunchFusedResourceApplyRMSPropKernel<T>(
          var_t.flat<T>().data(), ms_t.flat<T>().data(), mom_t.flat<T>().data(),
          grad.flat<T>().data(), lr, rho, momentum, epsilon, n, stream);

      CheckLaunchAndSync(ctx, "ApplyRMSProp");
    }

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;
};

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

    std::vector<ScopedMutexUnlocker> locks;
    LockResourceVariables({var.get(), mg.get(), ms.get(), mom.get()}, &locks);

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    mg->tensor()->IsInitialized() &&
                    ms->tensor()->IsInitialized() &&
                    mom->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Fused centered RMSProp variables (var/mg/ms/mom) not "
                    "initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, mg.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, ms.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedRMSPropUpdate(ctx, mom.get()));

    Tensor* var_t = var->tensor();
    Tensor* mg_t = mg->tensor();
    Tensor* ms_t = ms->tensor();
    Tensor* mom_t = mom->tensor();
    const Tensor& grad = ctx->input(8);

    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *mg_t, "var", "mg"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *ms_t, "var", "ms"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *mom_t, "var", "mom"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "rho"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "momentum"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(7), "epsilon"));

    const int64_t n = var_t->NumElements();
    if (n == 0) return;

    const double lr = ScalarToDouble<T>(ctx->input(4));
    const double rho = ScalarToDouble<T>(ctx->input(5));
    const double momentum = ScalarToDouble<T>(ctx->input(6));
    const double epsilon = ScalarToDouble<T>(ctx->input(7));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchFusedResourceApplyCenteredRMSPropKernel<T>(
        var_t->flat<T>().data(), mg_t->flat<T>().data(), ms_t->flat<T>().data(),
        mom_t->flat<T>().data(), grad.flat<T>().data(), lr, rho, momentum,
        epsilon, n, stream);

    CheckLaunchAndSync(ctx, "ResourceApplyCenteredRMSProp");
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;
};

template <typename T>
class MusaApplyCenteredRMSPropKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyCenteredRMSPropKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<ScopedMutexUnlocker> locks;
    if (use_exclusive_lock_) {
      LockRefInputs(ctx, {0, 1, 2, 3}, &locks);
    }

    Tensor var_t = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor mg_t = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor ms_t = ctx->mutable_input(2, use_exclusive_lock_);
    Tensor mom_t = ctx->mutable_input(3, use_exclusive_lock_);
    const Tensor& grad = ctx->input(8);

    OP_REQUIRES(ctx,
                var_t.IsInitialized() && mg_t.IsInitialized() &&
                    ms_t.IsInitialized() && mom_t.IsInitialized(),
                errors::FailedPrecondition(
                    "Fused ApplyCenteredRMSProp variables (var/mg/ms/mom) not "
                    "initialized."));

    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, mg_t, "var", "mg"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, ms_t, "var", "ms"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, mom_t, "var", "mom"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "rho"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "momentum"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(7), "epsilon"));

    const int64_t n = var_t.NumElements();
    if (n != 0) {
      const double lr = ScalarToDouble<T>(ctx->input(4));
      const double rho = ScalarToDouble<T>(ctx->input(5));
      const double momentum = ScalarToDouble<T>(ctx->input(6));
      const double epsilon = ScalarToDouble<T>(ctx->input(7));

      musaStream_t stream = GetMusaStreamByCtx(ctx);
      LaunchFusedResourceApplyCenteredRMSPropKernel<T>(
          var_t.flat<T>().data(), mg_t.flat<T>().data(), ms_t.flat<T>().data(),
          mom_t.flat<T>().data(), grad.flat<T>().data(), lr, rho, momentum,
          epsilon, n, stream);

      CheckLaunchAndSync(ctx, "ApplyCenteredRMSProp");
    }

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_FUSED_RESOURCE_RMSPROP(T)                  \
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

#define REGISTER_FUSED_APPLY_RMSPROP(T)                         \
  REGISTER_KERNEL_BUILDER(Name("ApplyRMSProp")                  \
                              .Device(DEVICE_MTGPU)             \
                              .TypeConstraint<T>("T")           \
                              .HostMemory("lr")                 \
                              .HostMemory("rho")                \
                              .HostMemory("momentum")           \
                              .HostMemory("epsilon"),           \
                          MusaApplyRMSPropKernelOp<T>);

#define REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP(T)             \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyCenteredRMSProp")  \
                              .Device(DEVICE_MTGPU)             \
                              .HostMemory("var")                \
                              .HostMemory("mg")                 \
                              .HostMemory("ms")                 \
                              .HostMemory("mom")                \
                              .TypeConstraint<T>("T")           \
                              .HostMemory("lr")                 \
                              .HostMemory("rho")                \
                              .HostMemory("momentum")           \
                              .HostMemory("epsilon"),           \
                          MusaResourceApplyCenteredRMSPropOp<T>);

#define REGISTER_FUSED_APPLY_CENTERED_RMSPROP(T)                \
  REGISTER_KERNEL_BUILDER(Name("ApplyCenteredRMSProp")          \
                              .Device(DEVICE_MTGPU)             \
                              .TypeConstraint<T>("T")           \
                              .HostMemory("lr")                 \
                              .HostMemory("rho")                \
                              .HostMemory("momentum")           \
                              .HostMemory("epsilon"),           \
                          MusaApplyCenteredRMSPropKernelOp<T>);

REGISTER_FUSED_RESOURCE_RMSPROP(float);
REGISTER_FUSED_RESOURCE_RMSPROP(double);
REGISTER_FUSED_RESOURCE_RMSPROP(Eigen::half);
REGISTER_FUSED_RESOURCE_RMSPROP(bfloat16);

REGISTER_FUSED_APPLY_RMSPROP(float);
REGISTER_FUSED_APPLY_RMSPROP(double);
REGISTER_FUSED_APPLY_RMSPROP(Eigen::half);
REGISTER_FUSED_APPLY_RMSPROP(bfloat16);

REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP(float);
REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP(double);
REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP(Eigen::half);
REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP(bfloat16);

REGISTER_FUSED_APPLY_CENTERED_RMSPROP(float);
REGISTER_FUSED_APPLY_CENTERED_RMSPROP(double);
REGISTER_FUSED_APPLY_CENTERED_RMSPROP(Eigen::half);
REGISTER_FUSED_APPLY_CENTERED_RMSPROP(bfloat16);

#undef REGISTER_FUSED_RESOURCE_RMSPROP
#undef REGISTER_FUSED_APPLY_RMSPROP
#undef REGISTER_FUSED_RESOURCE_CENTERED_RMSPROP
#undef REGISTER_FUSED_APPLY_CENTERED_RMSPROP

}  // namespace musa
}  // namespace tensorflow
