#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

template <typename T>
class BroadcastGradientArgsOp : public OpKernel {
 public:
  explicit BroadcastGradientArgsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& s0 = ctx->input(0);
    const Tensor& s1 = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(s0.shape()),
                errors::InvalidArgument("Input 0 must be a vector."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(s1.shape()),
                errors::InvalidArgument("Input 1 must be a vector."));

    TensorShape shape0, shape1;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(s0.vec<T>(), &shape0));
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(s1.vec<T>(), &shape1));

    BCast bcast(BCast::FromShape(shape0), BCast::FromShape(shape1));

    if (!bcast.IsValid()) {
      ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: [",
                                             s0.SummarizeValue(10), "] vs. [",
                                             s1.SummarizeValue(10), "]"));
      return;
    }

    // Allocate outputs
    Tensor* r0 = nullptr;
    Tensor* r1 = nullptr;

    const size_t len0 = bcast.grad_x_reduce_idx().size();
    const size_t len1 = bcast.grad_y_reduce_idx().size();

    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, TensorShape({static_cast<int64_t>(len0)}), &r0));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1, TensorShape({static_cast<int64_t>(len1)}), &r1));

    auto r0_vec = r0->vec<T>();
    auto r1_vec = r1->vec<T>();

    for (size_t i = 0; i < len0; ++i) {
      r0_vec(i) = static_cast<T>(bcast.grad_x_reduce_idx()[i]);
    }
    for (size_t i = 0; i < len1; ++i) {
      r1_vec(i) = static_cast<T>(bcast.grad_y_reduce_idx()[i]);
    }
  }
};

#define REGISTER_MUSA_BCAST_GRAD_ARGS(type)              \
  REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")  \
                              .Device("MUSA")            \
                              .TypeConstraint<type>("T") \
                              .HostMemory("s0")          \
                              .HostMemory("s1")          \
                              .HostMemory("r0")          \
                              .HostMemory("r1"),         \
                          BroadcastGradientArgsOp<type>)

REGISTER_MUSA_BCAST_GRAD_ARGS(int32);
REGISTER_MUSA_BCAST_GRAD_ARGS(int64);

#undef REGISTER_MUSA_BCAST_GRAD_ARGS

}  // namespace musa
}  // namespace tensorflow
