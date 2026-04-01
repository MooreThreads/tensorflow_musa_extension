#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <list>
#include <type_traits>
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

// Declarations for the fused MUSA kernel launchers implemented in
// musa_resource_apply_nadam_kernel.mu
extern "C" {
void LaunchResourceApplyNadamFloat(float* var, float* m, float* v,
                                   float beta1_power, float beta2_power,
                                   float lr, float beta1, float beta2,
                                   float epsilon, const float* grad, int64_t n,
                                   musaStream_t stream);
void LaunchResourceApplyNadamDouble(double* var, double* m, double* v,
                                    double beta1_power, double beta2_power,
                                    double lr, double beta1, double beta2,
                                    double epsilon, const double* grad,
                                    int64_t n, musaStream_t stream);
void LaunchResourceApplyNadamHalf(void* var, void* m, void* v,
                                  float beta1_power, float beta2_power,
                                  float lr, float beta1, float beta2,
                                  float epsilon, const void* grad, int64_t n,
                                  musaStream_t stream);
void LaunchResourceApplyNadamBFloat16(void* var, void* m, void* v,
                                      float beta1_power, float beta2_power,
                                      float lr, float beta1, float beta2,
                                      float epsilon, const void* grad,
                                      int64_t n, musaStream_t stream);
}

namespace {
Status CopyTensorForUpdate(OpKernelContext* ctx, const Tensor& src,
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
    return errors::Internal("CopyTensorForUpdate: musaMemcpyAsync failed: ",
                            musaGetErrorString(err));
  }

  return Status::OK();
}

Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var) {
  if (!var->copy_on_read_mode.load() && var->tensor()->RefCountIsOne()) {
    return Status::OK();
  }

  Tensor copied;
  TF_RETURN_IF_ERROR(CopyTensorForUpdate(ctx, *var->tensor(), &copied));
  *var->tensor() = copied;
  return Status::OK();
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
}  // namespace

template <typename T>
class MusaResourceApplyNadamOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyNadamOp(OpKernelConstruction* ctx)
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

    // Call fused kernel launcher
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    const int64_t n = var_t.NumElements();
    if (n > 0) {
      if (std::is_same<T, float>::value) {
        LaunchResourceApplyNadamFloat(
            var_t.flat<float>().data(), m_t.flat<float>().data(),
            v_t.flat<float>().data(), static_cast<float>(beta1_power),
            static_cast<float>(beta2_power), static_cast<float>(lr),
            static_cast<float>(beta1), static_cast<float>(beta2),
            static_cast<float>(epsilon), grad.flat<float>().data(), n, stream);
      } else if (std::is_same<T, double>::value) {
        LaunchResourceApplyNadamDouble(
            var_t.flat<double>().data(), m_t.flat<double>().data(),
            v_t.flat<double>().data(), static_cast<double>(beta1_power),
            static_cast<double>(beta2_power), static_cast<double>(lr),
            static_cast<double>(beta1), static_cast<double>(beta2),
            static_cast<double>(epsilon), grad.flat<double>().data(), n,
            stream);
      } else if (std::is_same<T, Eigen::half>::value) {
        LaunchResourceApplyNadamHalf(
            const_cast<void*>(reinterpret_cast<const void*>(
                var_t.flat<Eigen::half>().data())),
            const_cast<void*>(
                reinterpret_cast<const void*>(m_t.flat<Eigen::half>().data())),
            const_cast<void*>(
                reinterpret_cast<const void*>(v_t.flat<Eigen::half>().data())),
            static_cast<float>(beta1_power), static_cast<float>(beta2_power),
            static_cast<float>(lr), static_cast<float>(beta1),
            static_cast<float>(beta2), static_cast<float>(epsilon),
            reinterpret_cast<const void*>(grad.flat<Eigen::half>().data()), n,
            stream);
      } else if (std::is_same<T, bfloat16>::value) {
        LaunchResourceApplyNadamBFloat16(
            const_cast<void*>(
                reinterpret_cast<const void*>(var_t.flat<bfloat16>().data())),
            const_cast<void*>(
                reinterpret_cast<const void*>(m_t.flat<bfloat16>().data())),
            const_cast<void*>(
                reinterpret_cast<const void*>(v_t.flat<bfloat16>().data())),
            static_cast<float>(beta1_power), static_cast<float>(beta2_power),
            static_cast<float>(lr), static_cast<float>(beta1),
            static_cast<float>(beta2), static_cast<float>(epsilon),
            reinterpret_cast<const void*>(grad.flat<bfloat16>().data()), n,
            stream);
      }
    }

    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyNadam: musaStreamSynchronize "
                                 "failed: ",
                                 musaGetErrorString(sync_err)));

    for (int i = 0; i < ctx->num_outputs(); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
  }

 private:
  bool use_exclusive_lock_;
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

#undef REGISTER_RESOURCE_NADAM

}  // namespace musa
}  // namespace tensorflow
