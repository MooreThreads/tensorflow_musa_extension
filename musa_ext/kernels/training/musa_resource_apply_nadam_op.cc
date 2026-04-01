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

  void UseMudnn(OpKernelContext* ctx, Tensor& var_t, Tensor& m_t, Tensor& v_t,
                const Tensor& grad, T beta1_power, T beta2_power, T lr, T beta1,
                T beta2, T epsilon) {
    auto& handle = GetHandleByCtx(ctx);
    std::list<Tensor> temp_storage;
    ::musa::dnn::Binary b_op;
    ::musa::dnn::Unary u_op;

    auto require_success = [&](::musa::dnn::Status status,
                               const char* op_name) -> Status {
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal("ResourceApplyNadam ", op_name,
                                " failed. Status: ", static_cast<int>(status));
      }
      return Status::OK();
    };

    auto fill_scalar = [&](T val, const TensorShape& shape,
                           mTensor* out) -> Status {
      temp_storage.emplace_back();
      Status alloc_status = ctx->allocate_temp(DataTypeToEnum<T>::value, shape,
                                               &temp_storage.back());
      if (!alloc_status.ok()) {
        return alloc_status;
      }
      *out = CreateMTensor(temp_storage.back(), format_);
      return MusaFillCall(out, val, ctx);
    };

    mTensor t_var = CreateMTensor(var_t, format_);
    mTensor t_m = CreateMTensor(m_t, format_);
    mTensor t_v = CreateMTensor(v_t, format_);
    mTensor t_grad = CreateMTensor(grad, format_);

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    mTensor t_beta1;
    mTensor t_one_minus_beta1;
    OP_REQUIRES_OK(ctx, fill_scalar(beta1, m_t.shape(), &t_beta1));
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta1, grad.shape(),
                                    &t_one_minus_beta1));

    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_m, t_m, t_beta1),
                                        "MUL m beta1"));

    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_g_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_g_scaled, t_grad,
                                                 t_one_minus_beta1),
                                        "MUL g one_minus_beta1"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_m, t_m, t_g_scaled), "ADD m"));

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    mTensor t_beta2;
    mTensor t_one_minus_beta2;
    OP_REQUIRES_OK(ctx, fill_scalar(beta2, v_t.shape(), &t_beta2));
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta2, grad.shape(),
                                    &t_one_minus_beta2));

    // v = v * beta2
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_v, t_v, t_beta2),
                                        "MUL v beta2"));

    // g^2
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_g2 = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_g2, t_grad, t_grad),
                                        "MUL g g"));

    // (1-beta2) * g^2
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           grad.shape(), &temp_storage.back()));
    mTensor t_g2_scaled = CreateMTensor(temp_storage.back(), format_);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_g2_scaled, t_g2,
                                                 t_one_minus_beta2),
                                        "MUL g2 scaled"));

    // v = v + (1-beta2)*g^2
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_v, t_v, t_g2_scaled), "ADD v"));

    // m_hat = (beta1 * m_t + (1 - beta1) * g_t) / (1 - beta1_power)
    // First calculate beta1 * m_t + (1 - beta1) * g_t
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           m_t.shape(), &temp_storage.back()));
    mTensor t_m_hat_num = CreateMTensor(temp_storage.back(), format_);

    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_m_hat_num, t_m, t_beta1),
                                   "MUL m_t beta1"));

    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_m_hat_num,
                                                 t_m_hat_num, t_g_scaled),
                                        "ADD m_hat_num"));

    // Divide by (1 - beta1_power)
    mTensor t_one_minus_beta1_power;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta1_power,
                                    m_t.shape(), &t_one_minus_beta1_power));
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_m_hat_num, t_m_hat_num,
                                            t_one_minus_beta1_power),
                                   "DIV m_hat"));

    // v_hat = v_t / (1 - beta2_power)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_v_hat = CreateMTensor(temp_storage.back(), format_);

    mTensor t_one_minus_beta2_power;
    OP_REQUIRES_OK(ctx, fill_scalar(static_cast<T>(1.0) - beta2_power,
                                    v_t.shape(), &t_one_minus_beta2_power));
    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_v_hat, t_v,
                                                 t_one_minus_beta2_power),
                                        "DIV v_hat"));

    // var_t = var_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    temp_storage.emplace_back();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           v_t.shape(), &temp_storage.back()));
    mTensor t_den = CreateMTensor(temp_storage.back(), format_);
    u_op.SetMode(::musa::dnn::Unary::Mode::SQRT);
    OP_REQUIRES_OK(
        ctx, require_success(u_op.Run(handle, t_den, t_v_hat), "SQRT v_hat"));

    mTensor t_eps;
    OP_REQUIRES_OK(ctx, fill_scalar(epsilon, v_t.shape(), &t_eps));
    b_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    OP_REQUIRES_OK(ctx, require_success(b_op.Run(handle, t_den, t_den, t_eps),
                                        "ADD epsilon"));

    b_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    OP_REQUIRES_OK(ctx,
                   require_success(b_op.Run(handle, t_den, t_m_hat_num, t_den),
                                   "DIV update"));

    mTensor t_lr;
    OP_REQUIRES_OK(ctx, fill_scalar(lr, var_t.shape(), &t_lr));
    b_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_den, t_den, t_lr), "MUL lr"));

    b_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    OP_REQUIRES_OK(
        ctx, require_success(b_op.Run(handle, t_var, t_var, t_den), "SUB var"));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(
        ctx, sync_err == musaSuccess,
        errors::Internal("ResourceApplyNadam: musaStreamSynchronize failed: ",
                         musaGetErrorString(sync_err)));
  }

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

}  // namespace musa
}  // namespace tensorflow
