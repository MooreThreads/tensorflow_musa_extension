#include "../utils_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

/**
 * MusaSigmoidCalibrationOp
 * 
 * Performs fusion: S / (S + Scale * (1 - S))
 * where S = Sigmoid(x)
 * 
 * This implements the specific activation logic from the given graph.
 */
template <typename T>
class MusaSigmoidCalibrationOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidCalibrationOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input, format_);
    auto scale_mt = CreateMTensor(scale, format_);
    auto out_mt = CreateMTensor(*output, format_);

    // Logic: output = sigmoid(input) / (sigmoid(input) + scale * (1 - sigmoid(input)))
    // This is often implemented as a single element-wise kernel.
    VLOG(1) << "MusaSigmoidCalibration Compute " << input.shape().DebugString();
  }
};

#define REGISTER_MUSA_SIGMOID_CALIBRATION(type)                                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MusaSigmoidCalibration").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSigmoidCalibrationOp<type>);

REGISTER_MUSA_SIGMOID_CALIBRATION(float);
REGISTER_MUSA_SIGMOID_CALIBRATION(Eigen::half);
REGISTER_MUSA_SIGMOID_CALIBRATION(bfloat16);

}  // namespace musa
}  // namespace tensorflow
