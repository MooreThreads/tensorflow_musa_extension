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
#include <vector>

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
extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

extern "C" {
void LaunchApplyAdaMaxFloat(float* var, float* m, float* v, const float* grad,
                            float beta1, float one_minus_beta1, float beta2,
                            float epsilon, float lr_t, int64_t n,
                            musaStream_t stream);
void LaunchApplyAdaMaxDouble(double* var, double* m, double* v,
                             const double* grad, double beta1,
                             double one_minus_beta1, double beta2,
                             double epsilon, double lr_t, int64_t n,
                             musaStream_t stream);
void LaunchApplyAdaMaxHalf(void* var, void* m, void* v, const void* grad,
                           float beta1, float one_minus_beta1, float beta2,
                           float epsilon, float lr_t, int64_t n,
                           musaStream_t stream);
void LaunchApplyAdaMaxBFloat16(void* var, void* m, void* v, const void* grad,
                               float beta1, float one_minus_beta1,
                               float beta2, float epsilon, float lr_t,
                               int64_t n, musaStream_t stream);
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

inline Status ValidateAdaMaxHyperParams(OpKernelContext* ctx) {
  const auto require_scalar = [&](int index, const char* name) -> Status {
    const Tensor& t = ctx->input(index);
    if (t.NumElements() != 1) {
      return errors::InvalidArgument(name, " must be a scalar, got shape ",
                                     t.shape().DebugString());
    }
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(require_scalar(3, "beta1_power"));
  TF_RETURN_IF_ERROR(require_scalar(4, "lr"));
  TF_RETURN_IF_ERROR(require_scalar(5, "beta1"));
  TF_RETURN_IF_ERROR(require_scalar(6, "beta2"));
  TF_RETURN_IF_ERROR(require_scalar(7, "epsilon"));
  return Status::OK();
}

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
struct AdaMaxLauncher;

template <>
struct AdaMaxLauncher<float> {
  static void Launch(Tensor* var_t, Tensor* m_t, Tensor* v_t,
                     const Tensor& grad_t, float beta1, float one_minus_beta1,
                     float beta2, float epsilon, float lr_t, int64_t n,
                     musaStream_t stream) {
    LaunchApplyAdaMaxFloat(var_t->flat<float>().data(), m_t->flat<float>().data(),
                           v_t->flat<float>().data(),
                           grad_t.flat<float>().data(), beta1, one_minus_beta1,
                           beta2, epsilon, lr_t, n, stream);
  }
};

template <>
struct AdaMaxLauncher<double> {
  static void Launch(Tensor* var_t, Tensor* m_t, Tensor* v_t,
                     const Tensor& grad_t, double beta1,
                     double one_minus_beta1, double beta2, double epsilon,
                     double lr_t, int64_t n, musaStream_t stream) {
    LaunchApplyAdaMaxDouble(
        var_t->flat<double>().data(), m_t->flat<double>().data(),
        v_t->flat<double>().data(), grad_t.flat<double>().data(), beta1,
        one_minus_beta1, beta2, epsilon, lr_t, n, stream);
  }
};

template <>
struct AdaMaxLauncher<Eigen::half> {
  static void Launch(Tensor* var_t, Tensor* m_t, Tensor* v_t,
                     const Tensor& grad_t, float beta1, float one_minus_beta1,
                     float beta2, float epsilon, float lr_t, int64_t n,
                     musaStream_t stream) {
    LaunchApplyAdaMaxHalf(
        reinterpret_cast<void*>(var_t->flat<Eigen::half>().data()),
        reinterpret_cast<void*>(m_t->flat<Eigen::half>().data()),
        reinterpret_cast<void*>(v_t->flat<Eigen::half>().data()),
        reinterpret_cast<const void*>(grad_t.flat<Eigen::half>().data()), beta1,
        one_minus_beta1, beta2, epsilon, lr_t, n, stream);
  }
};

template <>
struct AdaMaxLauncher<bfloat16> {
  static void Launch(Tensor* var_t, Tensor* m_t, Tensor* v_t,
                     const Tensor& grad_t, float beta1, float one_minus_beta1,
                     float beta2, float epsilon, float lr_t, int64_t n,
                     musaStream_t stream) {
    LaunchApplyAdaMaxBFloat16(
        reinterpret_cast<void*>(var_t->flat<bfloat16>().data()),
        reinterpret_cast<void*>(m_t->flat<bfloat16>().data()),
        reinterpret_cast<void*>(v_t->flat<bfloat16>().data()),
        reinterpret_cast<const void*>(grad_t.flat<bfloat16>().data()), beta1,
        one_minus_beta1, beta2, epsilon, lr_t, n, stream);
  }
};

template <typename T>
Status RunAdaMaxUpdate(OpKernelContext* ctx, Tensor* var_t, Tensor* m_t,
                       Tensor* v_t, const Tensor& grad_t, const T beta1_power,
                       const T lr, const T beta1, const T beta2,
                       const T epsilon) {
  const int64_t element_count = grad_t.NumElements();
  if (element_count == 0) {
    return Status::OK();
  }

  const double denom = 1.0 - static_cast<double>(beta1_power);
  if (denom == 0.0) {
    return errors::InvalidArgument("beta1_power must not be 1.");
  }

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  const T one_minus_beta1 = static_cast<T>(1) - beta1;
  const T lr_t = static_cast<T>(static_cast<double>(lr) / denom);

  AdaMaxLauncher<T>::Launch(var_t, m_t, v_t, grad_t, beta1, one_minus_beta1,
                            beta2, epsilon, lr_t, element_count, stream);

  musaError_t launch_err = musaGetLastError();
  if (launch_err != musaSuccess) {
    return errors::Internal("AdaMax kernel launch failed: ",
                            musaGetErrorString(launch_err));
  }

  musaError_t sync_err = musaStreamSynchronize(stream);
  if (sync_err != musaSuccess) {
    return errors::Internal("AdaMax musaStreamSynchronize failed: ",
                            musaGetErrorString(sync_err));
  }

  return Status::OK();
}

// ResourceApplyAdaMax Op using resource variables.
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
      if (mu != nullptr &&
          std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    };
    add_mutex(var->mu());
    add_mutex(m->mu());
    add_mutex(v->mu());
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

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    m->tensor()->IsInitialized() &&
                    v->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "AdaMax variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));
    OP_REQUIRES_OK(ctx, ValidateAdaMaxHyperParams(ctx));

    Tensor var_t = *var->tensor();
    Tensor m_t = *m->tensor();
    Tensor v_t = *v->tensor();
    const Tensor& grad_t = ctx->input(8);
    OP_REQUIRES_OK(ctx, ValidateAdaMaxShapes(var_t, m_t, v_t, grad_t));

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T lr = ctx->input(4).scalar<T>()();
    const T beta1 = ctx->input(5).scalar<T>()();
    const T beta2 = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();

    OP_REQUIRES_OK(ctx, RunAdaMaxUpdate(ctx, &var_t, &m_t, &v_t, grad_t,
                                        beta1_power, lr, beta1, beta2,
                                        epsilon));
  }

 private:
  bool use_exclusive_lock_;
};

// ApplyAdaMax Op using Ref tensors.
template <typename T>
class MusaApplyAdaMaxKernelOp : public MusaOpKernel {
 public:
  explicit MusaApplyAdaMaxKernelOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    std::vector<mutex*> mutexes;
    auto add_mutex = [&](mutex* mu) {
      if (mu != nullptr &&
          std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
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
    const Tensor& grad_t = ctx->input(8);

    OP_REQUIRES(ctx,
                var_t.IsInitialized() && m_t.IsInitialized() &&
                    v_t.IsInitialized(),
                errors::FailedPrecondition(
                    "AdaMax variables (var/m/v) not initialized."));
    OP_REQUIRES_OK(ctx, ValidateAdaMaxShapes(var_t, m_t, v_t, grad_t));

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T lr = ctx->input(4).scalar<T>()();
    const T beta1 = ctx->input(5).scalar<T>()();
    const T beta2 = ctx->input(6).scalar<T>()();
    const T epsilon = ctx->input(7).scalar<T>()();

    OP_REQUIRES_OK(ctx, RunAdaMaxUpdate(ctx, &var_t, &m_t, &v_t, grad_t,
                                        beta1_power, lr, beta1, beta2,
                                        epsilon));
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

REGISTER_RESOURCE_ADAMAX(float);
REGISTER_RESOURCE_ADAMAX(double);
REGISTER_RESOURCE_ADAMAX(Eigen::half);
REGISTER_RESOURCE_ADAMAX(bfloat16);

REGISTER_APPLY_ADAMAX(float);
REGISTER_APPLY_ADAMAX(double);
REGISTER_APPLY_ADAMAX(Eigen::half);
REGISTER_APPLY_ADAMAX(bfloat16);

#undef REGISTER_RESOURCE_ADAMAX
#undef REGISTER_APPLY_ADAMAX

}  // namespace musa
}  // namespace tensorflow
