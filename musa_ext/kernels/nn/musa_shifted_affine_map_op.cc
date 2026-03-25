#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaShiftedAffineMapOp : public MusaOpKernel {
 public:
  explicit MusaShiftedAffineMapOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    // TODO: Read attributes from ctx if needed
    // e.g. OP_REQUIRES_OK(ctx, ctx->GetAttr("some_attr", &some_attr_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "MusaShiftedAffineMap");

    // TODO: Implement the fused computation.
    //
    // Typical steps:
    //   1. Get input tensors
    //   2. Allocate output tensor
    //   3. Get muDNN handle and create mTensors
    //   4. Call muDNN API or launch custom .mu kernel
    //
    // Example:
    //   const Tensor& input = ctx->input(0);
    //   Tensor* output = nullptr;
    //   OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    //   if (input.NumElements() == 0) return;
    //   auto& handle = GetHandleByCtx(ctx);
    //   mTensor mt_input = CreateMTensor(input, format_);
    //   mTensor mt_output = CreateMTensor(*output, format_);
    //   // ... call muDNN op ...
  }

 private:
  // TODO: Add member variables for attributes
};

// Register kernel for supported types
#define REGISTER_MUSA_SHIFTED_AFFINE_MAP(TYPE)               \
  REGISTER_KERNEL_BUILDER(Name("MusaShiftedAffineMap")       \
                              .Device("MUSA")                \
                              .TypeConstraint<TYPE>("T"),     \
                          MusaShiftedAffineMapOp<TYPE>);

REGISTER_MUSA_SHIFTED_AFFINE_MAP(float);
REGISTER_MUSA_SHIFTED_AFFINE_MAP(Eigen::half);

#undef REGISTER_MUSA_SHIFTED_AFFINE_MAP

}  // namespace musa

// Register the Op definition
REGISTER_OP("MusaShiftedAffineMap")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, half}")
    // TODO: Add more inputs/outputs/attrs as needed for your fusion
    // e.g. .Input("weight: T")
    //      .Input("bias: T")
    //      .Attr("some_param: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // TODO: Implement proper shape inference
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
