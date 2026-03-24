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

template <typename T, typename Index>
struct AdaGradV2Launcher {
  static void Run(void* var, void* accum, const void* lr, const void* epsilon,
                  const void* grad, const Index* indices, int64_t inner_size,
                  int64_t indices_size, musaStream_t stream) {
    if constexpr (std::is_same<T, float>::value) {
      if constexpr (std::is_same<Index, int32>::value) {
        LaunchResourceSparseApplyAdaGradV2FloatInt32(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int32_t*>(indices), inner_size, indices_size,
            stream);
      } else {
        LaunchResourceSparseApplyAdaGradV2FloatInt64(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int64_t*>(indices), inner_size, indices_size,
            stream);
      }
    } else if constexpr (std::is_same<T, Eigen::half>::value) {
      if constexpr (std::is_same<Index, int32>::value) {
        LaunchResourceSparseApplyAdaGradV2HalfInt32(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int32_t*>(indices), inner_size, indices_size,
            stream);
      } else {
        LaunchResourceSparseApplyAdaGradV2HalfInt64(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int64_t*>(indices), inner_size, indices_size,
            stream);
      }
    } else if constexpr (std::is_same<T, bfloat16>::value) {
      if constexpr (std::is_same<Index, int32>::value) {
        LaunchResourceSparseApplyAdaGradV2BFloat16Int32(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int32_t*>(indices), inner_size, indices_size,
            stream);
      } else {
        LaunchResourceSparseApplyAdaGradV2BFloat16Int64(
            var, accum, lr, epsilon, grad,
            reinterpret_cast<const int64_t*>(indices), inner_size, indices_size,
            stream);
      }
    } else {
      static_assert(always_false<T>::value,
                    "Unsupported T for AdaGradV2 launcher");
    }
  }
};

template <typename T, typename Index>
class MusaResourceSparseApplyAdaGradV2Op : public MusaOpKernel {
 public:
  explicit MusaResourceSparseApplyAdaGradV2Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
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

    auto launch_v2 = [&](int start_idx, int count) {
      void* var_ptr = const_cast<void*>(
          static_cast<const void*>(var_tensor->flat<T>().data()));
      void* accum_ptr = const_cast<void*>(
          static_cast<const void*>(accum_tensor->flat<T>().data()));
      const void* lr_ptr = static_cast<const void*>(lr.flat<T>().data());
      const void* epsilon_ptr =
          static_cast<const void*>(epsilon.flat<T>().data());
      const void* grad_ptr = static_cast<const void*>(
          &grad.flat<T>().data()[start_idx * inner_size]);
      const Index* indices_ptr = &indices.flat<Index>()(start_idx);

      AdaGradV2Launcher<T, Index>::Run(var_ptr, accum_ptr, lr_ptr, epsilon_ptr,
                                       grad_ptr, indices_ptr, inner_size, count,
                                       stream);
    };

    launch_v2(0, indices_size);
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
