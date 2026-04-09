#include <type_traits>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

extern "C" {
#define DECLARE_V2_LAUNCHER(T)                                      \
  void LaunchResourceSparseApplyAdaGradV2##T##Int32(                \
      void* var, void* accum, const void* lr, const void* epsilon,  \
      const void* grad, const int32_t* indices, int64_t inner_size, \
      int64_t indices_size, musaStream_t stream);                   \
  void LaunchResourceSparseApplyAdaGradV2##T##Int64(                \
      void* var, void* accum, const void* lr, const void* epsilon,  \
      const void* grad, const int64_t* indices, int64_t inner_size, \
      int64_t indices_size, musaStream_t stream);

DECLARE_V2_LAUNCHER(Float)
DECLARE_V2_LAUNCHER(Half)
DECLARE_V2_LAUNCHER(BFloat16)
}

namespace tensorflow {
namespace musa {

template <typename T>
struct always_false : std::false_type {};

template <typename T, typename IndexT>
void LaunchResourceSparseApplyAdaGradV2Impl(T* var, T* accum, const T* lr,
                                            const T* epsilon, const T* grad,
                                            const IndexT* indices,
                                            int64_t inner_size,
                                            int64_t indices_size,
                                            musaStream_t stream);

template <typename T, typename Index>
class MusaResourceSparseApplyAdaGradV2Op : public MusaOpKernel {
 public:
  explicit MusaResourceSparseApplyAdaGradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));
    core::RefCountPtr<Var> accum;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));

    const Tensor& lr = ctx->input(2);
    const Tensor& epsilon = ctx->input(3);
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices is not a vector: ",
                                        indices.shape().DebugString()));
    OP_REQUIRES(ctx, grad.dims() > 0,
                errors::InvalidArgument("grad must be at least 1D: ",
                                        grad.shape().DebugString()));
    OP_REQUIRES(
        ctx, grad.dim_size(0) == indices.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of grad and indices must match. grad shape: ",
            grad.shape().DebugString(),
            ", indices shape: ", indices.shape().DebugString()));

    mutex_lock ml_var(*(var->mu()));
    mutex_lock ml_accum(*(accum->mu()));

    Tensor* var_tensor = var->tensor();
    Tensor* accum_tensor = accum->tensor();

    OP_REQUIRES(ctx, var_tensor->shape().IsSameSize(accum_tensor->shape()),
                errors::InvalidArgument(
                    "var and accum must have the same shape. var shape: ",
                    var_tensor->shape().DebugString(),
                    ", accum shape: ", accum_tensor->shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var_tensor->shape()),
                errors::InvalidArgument("var must be at least 1D: ",
                                        var_tensor->shape().DebugString()));

    const int64_t inner_size =
        var_tensor->shape().num_elements() / var_tensor->dim_size(0);
    const int64_t indices_size = indices.dim_size(0);

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    // For robustness we shall check if there exist duplicated indices. But for
    // now we just ignoring such cases to make the implementation simpler.

    LaunchResourceSparseApplyAdaGradV2Impl<T, Index>(
        var_tensor->flat<T>().data(), accum_tensor->flat<T>().data(),
        lr.flat<T>().data(), epsilon.flat<T>().data(),
        &grad.flat<T>().data()[0], &indices.flat<Index>()(0), inner_size,
        indices_size, stream);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradV2")           \
                              .Device("MUSA")                            \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int32>("Tindices"),        \
                          MusaResourceSparseApplyAdaGradV2Op<T, int32>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradV2")           \
                              .Device("MUSA")                            \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int64>("Tindices"),        \
                          MusaResourceSparseApplyAdaGradV2Op<T, int64>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(Eigen::half);
REGISTER_KERNELS(bfloat16);

}  // namespace musa
}  // namespace tensorflow
