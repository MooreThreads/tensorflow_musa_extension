#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

extern Status PrepareTensorForMusaUpdate(OpKernelContext* ctx, Var* var);

// ============================================================================
// Define new TensorFlow op for sparse Adam
// ============================================================================
REGISTER_OP("MusaResourceSparseApplyAdam")
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
    .Input("indices: Tindices")
    .Attr("T: {float, half, bfloat16}")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    });

// Custom RAII unlocker to avoid issues with TF's mutex_lock macro
class MutexUnlocker {
 public:
  explicit MutexUnlocker(mutex* mu) : mu_(mu) {}
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

template <typename T, typename IndexT>
extern void LaunchResourceSparseApplyAdamImpl(
    T* var, T* m, T* v, const T* grad, const IndexT* indices, const T* lr,
    const T* beta1, const T* beta2, const T* epsilon, const T* beta1_power,
    const T* beta2_power, int64_t inner_size, int64_t indices_size,
    musaStream_t stream);

template <typename T, typename IndexT>
class MusaResourceSparseApplyAdamOp : public MusaOpKernel {
 public:
  explicit MusaResourceSparseApplyAdamOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
  }

  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "[debug for timo] calling MusaResourceSparseApplyAdamOp";
    // Lookup resource variables
    core::RefCountPtr<Var> var;
    core::RefCountPtr<Var> m_var;
    core::RefCountPtr<Var> v_var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m_var));
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v_var));

    // Lock all variables (following Ftrl pattern)
    std::vector<Var*> vars = {var.get(), m_var.get(), v_var.get()};
    std::vector<mutex*> mutexes;
    for (auto* v : vars) {
      mutex* mu = v->mu();
      if (std::find(mutexes.begin(), mutexes.end(), mu) == mutexes.end()) {
        mutexes.push_back(mu);
      }
    }
    std::sort(mutexes.begin(), mutexes.end());

    std::vector<MutexUnlocker> locks;
    locks.reserve(mutexes.size());
    for (mutex* mu : mutexes) {
      mu->lock();
      locks.emplace_back(mu);
    }

    // Validate initialization
    OP_REQUIRES(ctx,
                var->tensor()->IsInitialized() &&
                    m_var->tensor()->IsInitialized() &&
                    v_var->tensor()->IsInitialized(),
                errors::FailedPrecondition(
                    "Sparse Adam variables (var/m/v) not initialized."));

    // Validate shapes match
    Tensor* var_tensor = var->tensor();
    Tensor* m_tensor = m_var->tensor();
    Tensor* v_tensor = v_var->tensor();

    OP_REQUIRES(
        ctx, var_tensor->shape().IsSameSize(m_tensor->shape()),
        errors::InvalidArgument("var and m must have the same shape. var: ",
                                var_tensor->shape().DebugString(),
                                " m: ", m_tensor->shape().DebugString()));
    OP_REQUIRES(
        ctx, var_tensor->shape().IsSameSize(v_tensor->shape()),
        errors::InvalidArgument("var and v must have the same shape. var: ",
                                var_tensor->shape().DebugString(),
                                " v: ", v_tensor->shape().DebugString()));

    // Prepare tensors for update (handle copy-on-write)
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, m_var.get()));
    OP_REQUIRES_OK(ctx, PrepareTensorForMusaUpdate(ctx, v_var.get()));

    // Refresh tensor pointers after potential copy
    var_tensor = var->tensor();
    m_tensor = m_var->tensor();
    v_tensor = v_var->tensor();

    // Get hyperparameters (host memory scalars)
    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);

    // Validate hyperparameter shapes
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power must be scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power must be scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr must be scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 must be scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 must be scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon must be scalar: ",
                                        epsilon.shape().DebugString()));

    // Validate indices and grad shapes
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be a vector: ",
                                        indices.shape().DebugString()));
    OP_REQUIRES(ctx, grad.dims() > 0,
                errors::InvalidArgument("grad must be at least 1D: ",
                                        grad.shape().DebugString()));
    OP_REQUIRES(ctx, grad.dim_size(0) == indices.dim_size(0),
                errors::InvalidArgument(
                    "grad and indices dimension 0 must match. grad: ",
                    grad.shape().DebugString(),
                    ", indices: ", indices.shape().DebugString()));

    // Validate var is at least 2D for sparse update
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_tensor->shape()),
                errors::InvalidArgument("var must be at least 1D: ",
                                        var_tensor->shape().DebugString()));

    // Compute sizes
    const int64_t inner_size =
        var_tensor->shape().num_elements() / var_tensor->dim_size(0);
    const int64_t indices_size = indices.dim_size(0);

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchResourceSparseApplyAdamImpl<T, IndexT>(
        var_tensor->flat<T>().data(), m_tensor->flat<T>().data(),
        v_tensor->flat<T>().data(), grad.flat<T>().data(),
        indices.flat<IndexT>().data(), lr.flat<T>().data(),
        beta1.flat<T>().data(), beta2.flat<T>().data(),
        epsilon.flat<T>().data(), beta1_power.flat<T>().data(),
        beta2_power.flat<T>().data(), inner_size, indices_size, stream);

    musaError_t sync_err = musaStreamSynchronize(stream);
    OP_REQUIRES(
        ctx, sync_err == musaSuccess,
        errors::Internal(
            "MusaResourceSparseApplyAdam: musaStreamSynchronize failed: ",
            musaGetErrorString(sync_err)));
  }

 private:
  bool use_locking_;
};

// Register kernels for all dtype combinations
#define REGISTER_KERNELS(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("MusaResourceSparseApplyAdam")       \
                              .Device(DEVICE_MTGPU)                 \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tindices"),   \
                          MusaResourceSparseApplyAdamOp<T, int32>); \
  REGISTER_KERNEL_BUILDER(Name("MusaResourceSparseApplyAdam")       \
                              .Device(DEVICE_MTGPU)                 \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64>("Tindices"),   \
                          MusaResourceSparseApplyAdamOp<T, int64>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(Eigen::half);
REGISTER_KERNELS(bfloat16);

#undef REGISTER_KERNELS

}  // namespace musa
}  // namespace tensorflow
