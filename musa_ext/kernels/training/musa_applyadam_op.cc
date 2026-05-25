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
void LaunchFusedApplyAdamKernel(T* var, T* m, T* v, const T* grad,
                                double alpha, double beta1, double beta2,
                                double epsilon, bool use_nesterov, int64_t n,
                                musaStream_t stream);

Status CopyTensorForUpdate(OpKernelContext* ctx, const Tensor& src,
                           Tensor* dst) {
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  attr.set_nic_compatible(true);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst, attr));

  if (src.TotalBytes() == 0) return OkStatus();

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(dst->data(), src.data(), src.TotalBytes(),
                                    musaMemcpyDeviceToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal("CopyTensorForUpdate: musaMemcpyAsync failed: ",
                            musaGetErrorString(err));
  }
  return OkStatus();
}

Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return OkStatus();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(CopyTensorForUpdate(ctx, *var->tensor(), &copied));
  *var->tensor() = copied;
  return OkStatus();
}

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

double BiasCorrectedLearningRate(double beta1_power, double beta2_power,
                                 double lr) {
  const double one_minus_beta1_power = 1.0 - beta1_power;
  if (std::abs(one_minus_beta1_power) < 1e-10) {
    return lr;
  }
  return lr * std::sqrt(1.0 - beta2_power) / one_minus_beta1_power;
}

void CheckLaunch(OpKernelContext* ctx, const char* op_name) {
  musaError_t launch_err = musaGetLastError();
  OP_REQUIRES(ctx, launch_err == musaSuccess,
              errors::Internal(op_name, ": MUSA kernel launch failed: ",
                               musaGetErrorString(launch_err)));
}

}  // namespace

template <typename T>
class MusaResourceApplyAdamSingleKernelOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyAdamSingleKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> m;
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));

    std::vector<ScopedMutexUnlocker> locks;
    LockResourceVariables({var.get(), m.get(), v.get()}, &locks);

    OP_REQUIRES(
        ctx,
        var->tensor()->IsInitialized() && m->tensor()->IsInitialized() &&
            v->tensor()->IsInitialized(),
        errors::FailedPrecondition(
            "Adam variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));

    Tensor* var_t = var->tensor();
    Tensor* m_t = m->tensor();
    Tensor* v_t = v->tensor();
    const Tensor& grad = ctx->input(9);

    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *m_t, "var", "m"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, *v_t, "var", "v"));
    OP_REQUIRES_OK(ctx, CheckSameShape(*var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(3), "beta1_power"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "beta2_power"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "beta1"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(7), "beta2"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(8), "epsilon"));

    const int64_t n = var_t->NumElements();
    if (n == 0) return;

    const double beta1_power = ScalarToDouble<T>(ctx->input(3));
    const double beta2_power = ScalarToDouble<T>(ctx->input(4));
    const double lr = ScalarToDouble<T>(ctx->input(5));
    const double beta1 = ScalarToDouble<T>(ctx->input(6));
    const double beta2 = ScalarToDouble<T>(ctx->input(7));
    const double epsilon = ScalarToDouble<T>(ctx->input(8));
    const double alpha =
        BiasCorrectedLearningRate(beta1_power, beta2_power, lr);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchFusedApplyAdamKernel<T>(
        var_t->flat<T>().data(), m_t->flat<T>().data(), v_t->flat<T>().data(),
        grad.flat<T>().data(), alpha, beta1, beta2, epsilon, use_nesterov_, n,
        stream);
    CheckLaunch(ctx, "ResourceApplyAdam");
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

template <typename T>
class MusaApplyAdamSingleKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdamSingleKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    std::vector<ScopedMutexUnlocker> locks;
    if (use_exclusive_lock_) {
      LockRefInputs(ctx, {0, 1, 2}, &locks);
    }

    Tensor var_t = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor m_t = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor v_t = ctx->mutable_input(2, use_exclusive_lock_);
    const Tensor& grad = ctx->input(9);

    OP_REQUIRES(ctx,
                var_t.IsInitialized() && m_t.IsInitialized() &&
                    v_t.IsInitialized(),
                errors::FailedPrecondition(
                    "ApplyAdam variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, m_t, "var", "m"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, v_t, "var", "v"));
    OP_REQUIRES_OK(ctx, CheckSameShape(var_t, grad, "var", "grad"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(3), "beta1_power"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(4), "beta2_power"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(5), "lr"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(6), "beta1"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(7), "beta2"));
    OP_REQUIRES_OK(ctx, CheckScalar(ctx->input(8), "epsilon"));

    const int64_t n = var_t.NumElements();
    if (n != 0) {
      const double beta1_power = ScalarToDouble<T>(ctx->input(3));
      const double beta2_power = ScalarToDouble<T>(ctx->input(4));
      const double lr = ScalarToDouble<T>(ctx->input(5));
      const double beta1 = ScalarToDouble<T>(ctx->input(6));
      const double beta2 = ScalarToDouble<T>(ctx->input(7));
      const double epsilon = ScalarToDouble<T>(ctx->input(8));
      const double alpha =
          BiasCorrectedLearningRate(beta1_power, beta2_power, lr);

      musaStream_t stream = GetMusaStreamByCtx(ctx);
      LaunchFusedApplyAdamKernel<T>(
          var_t.flat<T>().data(), m_t.flat<T>().data(), v_t.flat<T>().data(),
          grad.flat<T>().data(), alpha, beta1, beta2, epsilon, use_nesterov_, n,
          stream);
      CheckLaunch(ctx, "ApplyAdam");
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
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
                          MusaResourceApplyAdamSingleKernelOp<T>);

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
                          MusaApplyAdamSingleKernelOp<T>);

REGISTER_RESOURCE_ADAM(float);
REGISTER_RESOURCE_ADAM(double);
REGISTER_RESOURCE_ADAM(Eigen::half);
REGISTER_RESOURCE_ADAM(bfloat16);

REGISTER_APPLY_ADAM(float);
REGISTER_APPLY_ADAM(double);
REGISTER_APPLY_ADAM(Eigen::half);
REGISTER_APPLY_ADAM(bfloat16);

#undef REGISTER_RESOURCE_ADAM
#undef REGISTER_APPLY_ADAM

}  // namespace musa
}  // namespace tensorflow
