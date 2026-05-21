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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchFusedApplyAdagradV2Kernel(T* var, T* accum, const T* grad,
                                     double lr, double epsilon,
                                     bool update_slots, int64_t n,
                                     musaStream_t stream);

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

Status CopyTensorForFusedAdagradV2Update(OpKernelContext* ctx,
                                         const Tensor& src, Tensor* dst) {
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  attr.set_nic_compatible(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst, attr));

  if (src.TotalBytes() == 0) return OkStatus();

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(dst->data(), src.data(), src.TotalBytes(),
                                    musaMemcpyDeviceToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal(
        "CopyTensorForFusedAdagradV2Update: musaMemcpyAsync failed: ",
        musaGetErrorString(err));
  }
  return OkStatus();
}

Status PrepareTensorForFusedAdagradV2Update(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return OkStatus();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(
      CopyTensorForFusedAdagradV2Update(ctx, *var->tensor(), &copied));
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

Status CheckSameShape(const Tensor& lhs, const Tensor& rhs,
                      const char* lhs_name, const char* rhs_name) {
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
class MusaFusedResourceApplyAdagradV2Op : public MusaOpKernel {
 public:
  explicit MusaFusedResourceApplyAdagradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> accum;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));

    std::vector<ScopedMutexUnlocker> locks;
    LockResourceVariables({var.get(), accum.get()}, &locks);

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    accum->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Fused AdagradV2 variables (var/accum) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForFusedAdagradV2Update(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForFusedAdagradV2Update(ctx, accum.get()));

    Tensor* var_t = var->tensor();
    Tensor* accum_t = accum->tensor();
    const Tensor& lr_t = ctx->input(2);
    const Tensor& epsilon_t = ctx->input(3);
    const Tensor& grad_t = ctx->input(4);

    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *accum_t, "var", "accum"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, grad_t, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(lr_t, "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(epsilon_t, "epsilon"));

    const int64_t n = var_t->NumElements();
    if (n == 0) return;

    const double lr = ScalarToDouble<T>(lr_t);
    const double epsilon = ScalarToDouble<T>(epsilon_t);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchFusedApplyAdagradV2Kernel<T>(
        var_t->flat<T>().data(), accum_t->flat<T>().data(),
        grad_t.flat<T>().data(), lr, epsilon, update_slots_, n, stream);

    CheckLaunchAndSync(ctx, "MusaFusedResourceApplyAdagradV2");
  }

  bool IsExpensive() override { return true; }

 private:
  bool update_slots_;
};

template <typename T>
class MusaFusedApplyAdagradV2Op : public MusaOpKernel {
 public:
  explicit MusaFusedApplyAdagradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override {
    std::vector<ScopedMutexUnlocker> locks;
    if (use_exclusive_lock_) {
      LockRefInputs(ctx, {0, 1}, &locks);
    }

    Tensor var_t = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum_t = ctx->mutable_input(1, use_exclusive_lock_);
    const Tensor& lr_t = ctx->input(2);
    const Tensor& epsilon_t = ctx->input(3);
    const Tensor& grad_t = ctx->input(4);

    OP_REQUIRES(ctx, var_t.IsInitialized() && accum_t.IsInitialized(),
                errors::FailedPrecondition(
                    "Fused ApplyAdagradV2 variables (var/accum) not "
                    "initialized."));

    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, accum_t, "var", "accum"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, grad_t, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(lr_t, "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(epsilon_t, "epsilon"));

    const int64_t n = var_t.NumElements();
    if (n != 0) {
      const double lr = ScalarToDouble<T>(lr_t);
      const double epsilon = ScalarToDouble<T>(epsilon_t);

      musaStream_t stream = GetMusaStreamByCtx(ctx);
      LaunchFusedApplyAdagradV2Kernel<T>(
          var_t.flat<T>().data(), accum_t.flat<T>().data(),
          grad_t.flat<T>().data(), lr, epsilon, update_slots_, n, stream);

      CheckLaunchAndSync(ctx, "MusaFusedApplyAdagradV2");
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->forward_ref_input_to_ref_output(1, 1);
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_FUSED_RESOURCE_ADAGRAD_V2(T)                   \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradV2")        \
                              .Device(DEVICE_MTGPU)             \
                              .HostMemory("var")                \
                              .HostMemory("accum")              \
                              .HostMemory("lr")                 \
                              .HostMemory("epsilon")            \
                              .TypeConstraint<T>("T"),          \
                          MusaFusedResourceApplyAdagradV2Op<T>);

#define REGISTER_FUSED_APPLY_ADAGRAD_V2(T)                      \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdagradV2")                \
                              .Device(DEVICE_MTGPU)             \
                              .HostMemory("lr")                 \
                              .HostMemory("epsilon")            \
                              .TypeConstraint<T>("T"),          \
                          MusaFusedApplyAdagradV2Op<T>);

REGISTER_FUSED_RESOURCE_ADAGRAD_V2(float);
REGISTER_FUSED_RESOURCE_ADAGRAD_V2(double);
REGISTER_FUSED_RESOURCE_ADAGRAD_V2(Eigen::half);
REGISTER_FUSED_RESOURCE_ADAGRAD_V2(bfloat16);

REGISTER_FUSED_APPLY_ADAGRAD_V2(float);
REGISTER_FUSED_APPLY_ADAGRAD_V2(double);
REGISTER_FUSED_APPLY_ADAGRAD_V2(Eigen::half);
REGISTER_FUSED_APPLY_ADAGRAD_V2(bfloat16);

#undef REGISTER_FUSED_RESOURCE_ADAGRAD_V2
#undef REGISTER_FUSED_APPLY_ADAGRAD_V2

}  // namespace musa
}  // namespace tensorflow
