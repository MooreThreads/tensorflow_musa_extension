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

// Mixed-precision dense ApplyAdam.
//
// var, m, v are required to be DT_FLOAT resource variables. grad is a dense
// tensor that may be float, half, or bfloat16. The full Adam update runs in
// fp32 inside the kernel; only the gradient narrows on load (with RNE
// rounding via the MUSA SDK intrinsics).
//
// This op exists because TensorFlow 2.6.1's ResourceApplyAdam ties var/m/v
// and grad to a single dtype T, which forces either:
//   * the math-in-bf16 mode (the legacy bf16 MusaResourceApplyAdam<bfloat16>
//     path), or
//   * an explicit Cast(bf16 -> fp32) ahead of fp32 ResourceApplyAdam, which
//     materializes a transient fp32 gradient on device.
// MusaResourceApplyAdamMixed avoids both.

#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

extern "C" {
void LaunchApplyAdamMixed_Float(float* var, float* m, float* v,
                                const void* grad, float lr_t, float beta1,
                                float beta2, float epsilon, int64_t n,
                                bool use_nesterov, musaStream_t stream);
void LaunchApplyAdamMixed_Half(float* var, float* m, float* v, const void* grad,
                               float lr_t, float beta1, float beta2,
                               float epsilon, int64_t n, bool use_nesterov,
                               musaStream_t stream);
void LaunchApplyAdamMixed_BFloat16(float* var, float* m, float* v,
                                   const void* grad, float lr_t, float beta1,
                                   float beta2, float epsilon, int64_t n,
                                   bool use_nesterov, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// PrepareTensorForMusaUpdate lives in musa_applyadam_op.cc; reuse it so dense
// Adam, sparse Adam, and this mixed-precision variant share copy-on-write
// semantics.
extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

namespace {

class MutexUnlockerMixed {
 public:
  explicit MutexUnlockerMixed(mutex* mu) : mu_(mu) {}
  MutexUnlockerMixed(MutexUnlockerMixed&& other) noexcept : mu_(other.mu_) {
    other.mu_ = nullptr;
  }
  MutexUnlockerMixed(const MutexUnlockerMixed&) = delete;
  MutexUnlockerMixed& operator=(const MutexUnlockerMixed&) = delete;
  ~MutexUnlockerMixed() {
    if (mu_ != nullptr) mu_->unlock();
  }

 private:
  mutex* mu_;
};

template <typename GradT>
struct AdamMixedLauncher;

template <>
struct AdamMixedLauncher<float> {
  static void Launch(float* var, float* m, float* v, const void* grad,
                     float lr_t, float beta1, float beta2, float epsilon,
                     int64_t n, bool use_nesterov, musaStream_t stream) {
    LaunchApplyAdamMixed_Float(var, m, v, grad, lr_t, beta1, beta2, epsilon, n,
                               use_nesterov, stream);
  }
};

template <>
struct AdamMixedLauncher<Eigen::half> {
  static void Launch(float* var, float* m, float* v, const void* grad,
                     float lr_t, float beta1, float beta2, float epsilon,
                     int64_t n, bool use_nesterov, musaStream_t stream) {
    LaunchApplyAdamMixed_Half(var, m, v, grad, lr_t, beta1, beta2, epsilon, n,
                              use_nesterov, stream);
  }
};

template <>
struct AdamMixedLauncher<bfloat16> {
  static void Launch(float* var, float* m, float* v, const void* grad,
                     float lr_t, float beta1, float beta2, float epsilon,
                     int64_t n, bool use_nesterov, musaStream_t stream) {
    LaunchApplyAdamMixed_BFloat16(var, m, v, grad, lr_t, beta1, beta2, epsilon,
                                  n, use_nesterov, stream);
  }
};

}  // namespace

template <typename GradT>
class MusaResourceApplyAdamMixedOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyAdamMixedOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    // 1. Resource lookup.
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> m;
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));

    // 2. Dedup mutexes and lock in deterministic order.
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
    for (mutex* mu : mutexes) mu->lock();
    std::vector<MutexUnlockerMixed> unlock_guards;
    unlock_guards.reserve(mutexes.size());
    for (mutex* mu : mutexes) unlock_guards.emplace_back(mu);

    // 3. Initialization check.
    OP_REQUIRES(
        ctx,
        var->tensor()->IsInitialized() && m->tensor()->IsInitialized() &&
            v->tensor()->IsInitialized(),
        errors::FailedPrecondition(
            "Mixed-precision Adam variables (var/m/v) not initialized."));

    // 4. dtype check: state must be fp32.
    OP_REQUIRES(ctx, var->tensor()->dtype() == DT_FLOAT,
                errors::InvalidArgument(
                    "MusaResourceApplyAdamMixed requires fp32 var, got ",
                    DataTypeString(var->tensor()->dtype())));
    OP_REQUIRES(ctx, m->tensor()->dtype() == DT_FLOAT,
                errors::InvalidArgument(
                    "MusaResourceApplyAdamMixed requires fp32 m, got ",
                    DataTypeString(m->tensor()->dtype())));
    OP_REQUIRES(ctx, v->tensor()->dtype() == DT_FLOAT,
                errors::InvalidArgument(
                    "MusaResourceApplyAdamMixed requires fp32 v, got ",
                    DataTypeString(v->tensor()->dtype())));

    // 5. Shape check.
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

    // 6. Copy-on-write handling (matches dense ResourceApplyAdam).
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));

    var_t = *var->tensor();
    m_t = *m->tensor();
    v_t = *v->tensor();

    // 7. Read host scalars (always fp32 per the op definition).
    const float beta1_power = ctx->input(3).scalar<float>()();
    const float beta2_power = ctx->input(4).scalar<float>()();
    const float lr = ctx->input(5).scalar<float>()();
    const float beta1 = ctx->input(6).scalar<float>()();
    const float beta2 = ctx->input(7).scalar<float>()();
    const float epsilon = ctx->input(8).scalar<float>()();

    // 8. Bias-corrected learning rate. Promote to fp64 for the division so
    // the first iteration (when (1 - beta1^t) is small) doesn't blow up.
    double lr_t_d;
    const double one_minus_b1p = 1.0 - static_cast<double>(beta1_power);
    if (std::abs(one_minus_b1p) < 1e-10) {
      lr_t_d = static_cast<double>(lr);
    } else {
      lr_t_d = static_cast<double>(lr) *
               std::sqrt(1.0 - static_cast<double>(beta2_power)) /
               one_minus_b1p;
    }
    const float lr_t = static_cast<float>(lr_t_d);

    if (var_t.NumElements() == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    AdamMixedLauncher<GradT>::Launch(
        var_t.flat<float>().data(), m_t.flat<float>().data(),
        v_t.flat<float>().data(),
        static_cast<const void*>(grad.tensor_data().data()), lr_t, beta1, beta2,
        epsilon, var_t.NumElements(), use_nesterov_, stream);

    // Mirror the legacy ResourceApplyAdam path's end-of-step sync. TODO: this
    // is a perf hazard in tight training loops; drop once we audit downstream
    // ordering assumptions.
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(
        ctx, sync_err == musaSuccess,
        errors::Internal(
            "MusaResourceApplyAdamMixed: musaStreamSynchronize failed: ",
            musaGetErrorString(sync_err)));
  }

 private:
  bool use_locking_ = false;
  bool use_nesterov_ = false;
};

}  // namespace musa

// ============================================================================
// REGISTER_OP
// ============================================================================
REGISTER_OP("MusaResourceApplyAdamMixed")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: float")
    .Input("beta2_power: float")
    .Input("lr: float")
    .Input("beta1: float")
    .Input("beta2: float")
    .Input("epsilon: float")
    .Input("grad: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return OkStatus();
    });

namespace musa {

#define REGISTER_ADAM_MIXED(GradT)                           \
  REGISTER_KERNEL_BUILDER(Name("MusaResourceApplyAdamMixed") \
                              .Device(DEVICE_MTGPU)          \
                              .TypeConstraint<GradT>("T")    \
                              .HostMemory("var")             \
                              .HostMemory("m")               \
                              .HostMemory("v")               \
                              .HostMemory("beta1_power")     \
                              .HostMemory("beta2_power")     \
                              .HostMemory("lr")              \
                              .HostMemory("beta1")           \
                              .HostMemory("beta2")           \
                              .HostMemory("epsilon"),        \
                          MusaResourceApplyAdamMixedOp<GradT>);

REGISTER_ADAM_MIXED(float);
REGISTER_ADAM_MIXED(Eigen::half);
REGISTER_ADAM_MIXED(bfloat16);

#undef REGISTER_ADAM_MIXED

}  // namespace musa
}  // namespace tensorflow
