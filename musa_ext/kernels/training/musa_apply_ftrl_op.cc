#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

// Custom RAII unlocker to avoid issues with TF's mutex_lock macro
class MutexUnlocker {
 public:
  explicit MutexUnlocker(mutex* mu) : mu_(mu) {}
  // Allow move but delete copy
  MutexUnlocker(MutexUnlocker&& other) noexcept : mu_(other.mu_) {
    other.mu_ = nullptr;
  }
  MutexUnlocker(const MutexUnlocker&) = delete;
  MutexUnlocker& operator=(const MutexUnlocker&) = delete;
  
  ~MutexUnlocker() {
    if (mu_ != nullptr) {
      mu_->unlock();
    }
  }

 private:
  mutex* mu_;
};

// Dense Launcher
template <typename T>
extern void LaunchApplyFtrlImpl(T* var, T* accum, T* linear, const T* grad,
                                const T* lr, const T* l1, const T* l2,
                                const T* l2_shrinkage, const T* lr_power,
                                int64_t total_elements, musaStream_t stream);

// Sparse Launcher
template <typename T, typename IndexT>
extern void LaunchResourceSparseApplyFtrlImpl(
    T* var, T* accum, T* linear, const T* grad, const IndexT* indices,
    const T* lr, const T* l1, const T* l2, const T* l2_shrinkage,
    const T* lr_power, int64_t inner_size, int64_t indices_size,
    musaStream_t stream);

// Helper to lock multiple resource variables
void LockResourceVariables(OpKernelContext* ctx, std::vector<Var*>& vars,
                           std::vector<MutexUnlocker>& locks) {
  std::vector<mutex*> mutexes;
  for (auto* var : vars) {
    mutex* mu = var->mu();
    if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
      mutexes.push_back(mu);
    }
  }
  std::sort(mutexes.begin(), mutexes.end());
  
  locks.reserve(mutexes.size());
  for (mutex* mu : mutexes) {
    mu->lock();
    locks.emplace_back(mu);
  }
}

template <typename T, bool HasShrinkage>
class MusaResourceApplyFtrlOp : public MusaOpKernel {
 public:
  explicit MusaResourceApplyFtrlOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> accum;
    core::RefCountPtr<Var> linear;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &linear));

    std::vector<Var*> vars = {var.get(), accum.get(), linear.get()};
    std::vector<MutexUnlocker> locks;
    LockResourceVariables(ctx, vars, locks);

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    accum->tensor()->IsInitialized() &&
                    linear->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Ftrl variables (var/accum/linear) not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, accum.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, linear.get()));

    Tensor* var_tensor = var->tensor();
    Tensor* accum_tensor = accum->tensor();
    Tensor* linear_tensor = linear->tensor();

    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& l1 = ctx->input(5);
    const Tensor& l2 = ctx->input(6);
    
    const Tensor* l2_shrinkage = nullptr;
    const Tensor* lr_power = nullptr;

    if (HasShrinkage) {
      l2_shrinkage = &ctx->input(7);
      lr_power = &ctx->input(8);
    } else {
      lr_power = &ctx->input(7);
    }

    OP_REQUIRES(ctx, var_tensor->shape().IsSameSize(grad.shape()),
                errors::InvalidArgument("var and grad must have same shape"));
    
    int64_t total_elements = var_tensor->shape().num_elements();
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchApplyFtrlImpl<T>(
        var_tensor->flat<T>().data(), accum_tensor->flat<T>().data(),
        linear_tensor->flat<T>().data(), grad.flat<T>().data(),
        lr.flat<T>().data(), l1.flat<T>().data(), l2.flat<T>().data(),
        HasShrinkage ? l2_shrinkage->flat<T>().data() : nullptr,
        lr_power->flat<T>().data(), total_elements, stream);

    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceApplyFtrl: musaStreamSynchronize failed"));
  }
};

template <typename T, bool HasShrinkage>
class MusaApplyFtrlOp : public MusaOpKernel {
 public:
  explicit MusaApplyFtrlOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor var;
    OP_REQUIRES_OK(ctx, ctx->mutable_input("var", &var, use_locking_));
    Tensor accum;
    OP_REQUIRES_OK(ctx, ctx->mutable_input("accum", &accum, use_locking_));
    Tensor linear;
    OP_REQUIRES_OK(ctx, ctx->mutable_input("linear", &linear, use_locking_));

    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& l1 = ctx->input(5);
    const Tensor& l2 = ctx->input(6);
    
    const Tensor* l2_shrinkage = nullptr;
    const Tensor* lr_power = nullptr;

    if (HasShrinkage) {
      l2_shrinkage = &ctx->input(7);
      lr_power = &ctx->input(8);
    } else {
      lr_power = &ctx->input(7);
    }

    int64_t total_elements = var.shape().num_elements();
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchApplyFtrlImpl<T>(
        var.flat<T>().data(), accum.flat<T>().data(),
        linear.flat<T>().data(), grad.flat<T>().data(),
        lr.flat<T>().data(), l1.flat<T>().data(), l2.flat<T>().data(),
        HasShrinkage ? l2_shrinkage->flat<T>().data() : nullptr,
        lr_power->flat<T>().data(), total_elements, stream);

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, var);
    }
  }
 private:
  bool use_locking_;
};

template <typename T, typename IndexT, bool HasShrinkage>
class MusaResourceSparseApplyFtrlOp : public MusaOpKernel {
 public:
  explicit MusaResourceSparseApplyFtrlOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> accum;
    core::RefCountPtr<Var> linear;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &linear));

    std::vector<Var*> vars = {var.get(), accum.get(), linear.get()};
    std::vector<MutexUnlocker> locks;
    LockResourceVariables(ctx, vars, locks);

    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    accum->tensor()->IsInitialized() &&
                    linear->tensor()->IsInitialized(),
                errors::FailedPrecondition("Ftrl variables not initialized."));

    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, accum.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, linear.get()));

    Tensor* var_tensor = var->tensor();
    Tensor* accum_tensor = accum->tensor();
    Tensor* linear_tensor = linear->tensor();

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& l1 = ctx->input(6);
    const Tensor& l2 = ctx->input(7);
    
    const Tensor* l2_shrinkage = nullptr;
    const Tensor* lr_power = nullptr;

    if (HasShrinkage) {
      l2_shrinkage = &ctx->input(8);
      lr_power = &ctx->input(9);
    } else {
      lr_power = &ctx->input(8);
    }

    const int64_t inner_size = var_tensor->shape().num_elements() / var_tensor->dim_size(0);
    const int64_t indices_size = indices.dim_size(0);

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchResourceSparseApplyFtrlImpl<T, IndexT>(
        var_tensor->flat<T>().data(), accum_tensor->flat<T>().data(),
        linear_tensor->flat<T>().data(), grad.flat<T>().data(),
        indices.flat<IndexT>().data(),
        lr.flat<T>().data(), l1.flat<T>().data(), l2.flat<T>().data(),
        HasShrinkage ? l2_shrinkage->flat<T>().data() : nullptr,
        lr_power->flat<T>().data(), inner_size, indices_size, stream);

    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(ctx, sync_err == musaSuccess,
                errors::Internal("ResourceSparseApplyFtrl: musaStreamSynchronize failed"));
  }
};

#define REGISTER_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyFtrl")                      \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T"),                   \
                          MusaResourceApplyFtrlOp<T, false>);            \
  REGISTER_KERNEL_BUILDER(Name("ApplyFtrl")                              \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T"),                   \
                          MusaApplyFtrlOp<T, false>);                    \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyFtrlV2")                    \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T"),                   \
                          MusaResourceApplyFtrlOp<T, true>);             \
  REGISTER_KERNEL_BUILDER(Name("ApplyFtrlV2")                            \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T"),                   \
                          MusaApplyFtrlOp<T, true>);                     \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyFtrl")                \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int32>("Tindices"),        \
                          MusaResourceSparseApplyFtrlOp<T, int32, false>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyFtrl")                \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int64>("Tindices"),        \
                          MusaResourceSparseApplyFtrlOp<T, int64, false>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyFtrlV2")              \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int32>("Tindices"),        \
                          MusaResourceSparseApplyFtrlOp<T, int32, true>);  \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyFtrlV2")              \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int64>("Tindices"),        \
                          MusaResourceSparseApplyFtrlOp<T, int64, true>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(Eigen::half);
REGISTER_KERNELS(bfloat16);

#undef REGISTER_KERNELS

}  // namespace musa
}  // namespace tensorflow
