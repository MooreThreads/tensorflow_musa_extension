#include <iostream>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLayerNormOp : public MusaOpKernel {
 public:
  explicit MusaLayerNormOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& gamma = ctx->input(1);
    const Tensor& beta = ctx->input(2);

    OP_REQUIRES(ctx, x.dims() >= 1,
                errors::InvalidArgument("Input rank must be >= 1"));

    int axis = x.dims() - 1;
    const int64 last_dim = x.dim_size(axis);

    OP_REQUIRES(
        ctx, gamma.NumElements() == last_dim,
        errors::InvalidArgument("Gamma size mismatch: expected ", last_dim,
                                ", got ", gamma.NumElements()));
    OP_REQUIRES(
        ctx, beta.NumElements() == last_dim,
        errors::InvalidArgument("Beta size mismatch: expected ", last_dim,
                                ", got ", beta.NumElements()));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    if (y->NumElements() == 0) return;

    TensorShape stat_shape;
    for (int i = 0; i < x.dims() - 1; ++i) {
      stat_shape.AddDim(x.dim_size(i));
    }

    Tensor mean;
    Tensor inv_var;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), stat_shape, &mean));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), stat_shape, &inv_var));

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_x = CreateMTensor(x, format_);
    mTensor mt_gamma = CreateMTensor(gamma, format_);
    mTensor mt_beta = CreateMTensor(beta, format_);
    mTensor mt_y = CreateMTensor(*y, format_);
    mTensor mt_mean = CreateMTensor(mean, format_);
    mTensor mt_inv_var = CreateMTensor(inv_var, format_);

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

    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status =
        ln.Run(handle, mt_y, mt_mean, mt_inv_var, mt_x, mt_gamma, mt_beta, mm);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA LayerNorm failed. Status=", (int)status));
  }

 private:
  float epsilon_;
};

#define REGISTER_MUSA_LAYERNORM(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MusaLayerNorm").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLayerNormOp<TYPE>);

REGISTER_MUSA_LAYERNORM(float);
REGISTER_MUSA_LAYERNORM(Eigen::half);
REGISTER_MUSA_LAYERNORM(bfloat16);

#undef REGISTER_MUSA_LAYERNORM

}  // namespace musa

REGISTER_OP("MusaLayerNorm")
    .Input("x: T")
    .Input("gamma: T")
    .Input("beta: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 0.00001")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow