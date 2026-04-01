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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchResourceApplyNadamKernel(T* var, T* m, T* v, const T* grad,
                                    float beta1_power, float beta2_power,
                                    float lr, float beta1, float beta2,
                                    float epsilon, int64_t n,
                                    musaStream_t stream);

// Helper functions for Resource variable updates
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

template <typename T>
class MusaResourceApplyNadamOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyNadamOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);
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
                    "Nadam variables (var/m/v) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v.get()));

    Tensor var_t = *var->tensor();
    Tensor m_t = *m->tensor();
    Tensor v_t = *v->tensor();

    const T beta1_power = ctx->input(3).scalar<T>()();
    const T beta2_power = ctx->input(4).scalar<T>()();
    const T lr = ctx->input(5).scalar<T>()();
    const T beta1 = ctx->input(6).scalar<T>()();
    const T beta2 = ctx->input(7).scalar<T>()();
    const T epsilon = ctx->input(8).scalar<T>()();
    const Tensor& grad = ctx->input(9);

    MUSA_KERNEL_TRACE_START("NadamKernel");
    // UseMudnn(ctx, var_t, m_t, v_t, grad, beta1_power, beta2_power, lr, beta1,
    //          beta2, epsilon);
    UseKernel(ctx, var_t, m_t, v_t, grad, beta1_power, beta2_power, lr, beta1,
              beta2, epsilon);
    MUSA_KERNEL_TRACE_END("NadamKernel");
  }

  bool IsExpensive() override { return true; }

 private:
  bool use_exclusive_lock_;

  // void UseMudnn(OpKernelContext* ctx, Tensor& var_t, Tensor& m_t, Tensor&
  // v_t,
  //               const Tensor& grad, T beta1_power, T beta2_power, T lr, T
  //               beta1, T beta2, T epsilon) {}

  void UseKernel(OpKernelContext* ctx, Tensor& var_t, Tensor& m_t, Tensor& v_t,
                 const Tensor& grad, T beta1_power, T beta2_power, T lr,
                 T beta1, T beta2, T epsilon) {
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchResourceApplyNadamKernel<T>(
        var_t.flat<T>().data(), m_t.flat<T>().data(), v_t.flat<T>().data(),
        grad.flat<T>().data(), static_cast<float>(beta1_power),
        static_cast<float>(beta2_power), static_cast<float>(lr),
        static_cast<float>(beta1), static_cast<float>(beta2),
        static_cast<float>(epsilon), var_t.NumElements(), stream);
  }
};

#define REGISTER_RESOURCE_NADAM(T)                       \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyNadam")     \
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
                          MusaResourceApplyNadamOp<T>);

REGISTER_RESOURCE_NADAM(float);
REGISTER_RESOURCE_NADAM(double);
REGISTER_RESOURCE_NADAM(Eigen::half);
REGISTER_RESOURCE_NADAM(bfloat16);

REGISTER_OP("ResourceApplyNadam")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace musa
}  // namespace tensorflow
