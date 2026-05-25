#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLayerNormGradOp : public MusaOpKernel {
 public:
  explicit MusaLayerNormGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dy = ctx->input(0);
    const Tensor& x = ctx->input(1);
    const Tensor& gamma = ctx->input(2);
    const Tensor& beta = ctx->input(3);

    OP_REQUIRES(ctx, x.dims() >= 1,
                errors::InvalidArgument("Input rank must be >= 1"));
    OP_REQUIRES(ctx, dy.shape() == x.shape(),
                errors::InvalidArgument(
                    "dy and x must have the same shape: dy=",
                    dy.shape().DebugString(), ", x=", x.shape().DebugString()));

    const int axis = x.dims() - 1;
    const int64 last_dim = x.dim_size(axis);

    OP_REQUIRES(
        ctx, gamma.NumElements() == last_dim,
        errors::InvalidArgument("Gamma size mismatch: expected ", last_dim,
                                ", got ", gamma.NumElements()));
    OP_REQUIRES(
        ctx, beta.NumElements() == last_dim,
        errors::InvalidArgument("Beta size mismatch: expected ", last_dim,
                                ", got ", beta.NumElements()));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    Tensor* dgamma = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, gamma.shape(), &dgamma));
    Tensor* dbeta = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, beta.shape(), &dbeta));

    auto& handle = GetHandleByCtx(ctx);
    auto stream = GetMusaStreamByCtx(ctx);

    if (x.NumElements() == 0) {
      musaMemsetAsync(const_cast<char*>(dx->tensor_data().data()), 0,
                      dx->TotalBytes(), stream);
      musaMemsetAsync(const_cast<char*>(dgamma->tensor_data().data()), 0,
                      dgamma->TotalBytes(), stream);
      musaMemsetAsync(const_cast<char*>(dbeta->tensor_data().data()), 0,
                      dbeta->TotalBytes(), stream);
      return;
    }

    TensorShape stat_shape;
    for (int i = 0; i < x.dims() - 1; ++i) {
      stat_shape.AddDim(x.dim_size(i));
    }

    Tensor y;
    Tensor mean;
    Tensor inv_var;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), stat_shape, &mean));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), stat_shape, &inv_var));

    mTensor mt_x = CreateMTensor(x, format_);
    mTensor mt_gamma = CreateMTensor(gamma, format_);
    mTensor mt_beta = CreateMTensor(beta, format_);
    mTensor mt_y = CreateMTensor(y, format_);
    mTensor mt_mean = CreateMTensor(mean, format_);
    mTensor mt_inv_var = CreateMTensor(inv_var, format_);
    mTensor mt_dy = CreateMTensor(dy, format_);
    mTensor mt_dx = CreateMTensor(*dx, format_);
    mTensor mt_dgamma = CreateMTensor(*dgamma, format_);
    mTensor mt_dbeta = CreateMTensor(*dbeta, format_);

    ::musa::dnn::LayerNorm ln;
    ln.SetEpsilon(epsilon_);

    std::vector<int> axis_vec;
    axis_vec.push_back(axis);
    ln.SetAxis(static_cast<int>(axis_vec.size()), axis_vec.data());

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      return std::unique_ptr<void, std::function<void(void*)>>(
          ptr, [tf_allocator](void* p) {
            if (p) tf_allocator->DeallocateRaw(p);
          });
    };
    ::musa::dnn::MemoryMaintainer maintainer(alloc_func);

    mStatus status =
        ln.Run(handle, mt_y, mt_mean, mt_inv_var, mt_x, mt_gamma, mt_beta,
               maintainer);
    OP_REQUIRES(
        ctx, status == mStatus::SUCCESS,
        errors::Internal("MUSA LayerNorm forward recompute failed. Status=",
                         static_cast<int>(status)));

    status = ln.RunBwd(handle, mt_dx, mt_dgamma, mt_dbeta, mt_dy, mt_x, mt_mean,
                       mt_inv_var, mt_gamma, maintainer);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA LayerNorm backward failed. Status=",
                                 static_cast<int>(status)));
  }

 private:
  float epsilon_;
};

#define REGISTER_MUSA_LAYERNORM_GRAD(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MusaLayerNormGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLayerNormGradOp<TYPE>);

REGISTER_MUSA_LAYERNORM_GRAD(float);
REGISTER_MUSA_LAYERNORM_GRAD(Eigen::half);
REGISTER_MUSA_LAYERNORM_GRAD(bfloat16);

#undef REGISTER_MUSA_LAYERNORM_GRAD

}  // namespace musa

REGISTER_OP("MusaLayerNormGrad")
    .Input("dy: T")
    .Input("x: T")
    .Input("gamma: T")
    .Input("beta: T")
    .Output("dx: T")
    .Output("dgamma: T")
    .Output("dbeta: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 0.00001")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      c->set_output(1, c->input(2));
      c->set_output(2, c->input(3));
      return OkStatus();
    });

}  // namespace tensorflow
