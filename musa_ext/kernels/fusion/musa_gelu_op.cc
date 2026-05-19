#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils/logging.h"

extern "C" {
void LaunchMusaGeluGradFloat(const float* x, const float* dy, float* dx,
                             int n, musaStream_t stream);
void LaunchMusaGeluGradBFloat16(const tensorflow::bfloat16* x,
                                const tensorflow::bfloat16* dy,
                                tensorflow::bfloat16* dx, int n,
                                musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGeluOp : public MusaOpKernel {
 public:
  explicit MusaGeluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) {
      // VLOG(1) << "MusaGeluOp::Compute skipped empty tensor";
      return;
    }

    const int64 num_elements = input.NumElements();
    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);
    mUnary op;
    const UNARY_MODE mode =
        approximate_ ? UNARY_MODE::GELU_TANH : UNARY_MODE::GELU;

    VLOG(1) << "MusaGeluOp::Compute launching muDNN GELU, elements="
            << num_elements << ", approximate=" << approximate_
            << ", mode=" << (approximate_ ? "GELU_TANH" : "GELU");

    MTOP_CHECK_OK(op.SetMode(mode), "Set GELU Mode", ctx);

    MTOP_CHECK_OK_RUN(op.Run(handle, mt_output, mt_input), "GELU Forward Run",
                      ctx);

    // VLOG(1) << "MusaGeluOp::Compute finished, elements=" << num_elements
    //         << ", approximate=" << approximate_;
  }

 private:
  bool approximate_;
};

class MusaGeluGradOp : public MusaOpKernel {
 public:
  explicit MusaGeluGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& dy = ctx->input(1);
    OP_REQUIRES(ctx, x.dtype() == dy.dtype() &&
                         (x.dtype() == DT_FLOAT || x.dtype() == DT_BFLOAT16),
                errors::InvalidArgument(
                    "MusaGeluGrad expects float/bfloat16 inputs"));
    OP_REQUIRES(ctx, x.shape() == dy.shape(),
                errors::InvalidArgument("MusaGeluGrad x/dy shape mismatch"));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    const int64_t num_elements = x.NumElements();
    if (num_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
    if (x.dtype() == DT_FLOAT) {
      LaunchMusaGeluGradFloat(
          x.flat<float>().data(), dy.flat<float>().data(),
          dx->flat<float>().data(), static_cast<int>(num_elements), stream);
    } else {
      LaunchMusaGeluGradBFloat16(
          x.flat<bfloat16>().data(), dy.flat<bfloat16>().data(),
          dx->flat<bfloat16>().data(), static_cast<int>(num_elements), stream);
    }
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaGeluGrad kernel failed: ",
                                 musaGetErrorString(status)));
  }
};

#define REGISTER_MUSA_GELU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MusaGelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaGeluOp<TYPE>);

REGISTER_MUSA_GELU(float);
REGISTER_MUSA_GELU(double);
REGISTER_MUSA_GELU(Eigen::half);
REGISTER_MUSA_GELU(bfloat16);

#undef REGISTER_MUSA_GELU

#define REGISTER_MUSA_GELU_GRAD(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MusaGeluGrad").Device("MUSA").TypeConstraint<TYPE>("T"),  \
      MusaGeluGradOp);

REGISTER_MUSA_GELU_GRAD(float);
REGISTER_MUSA_GELU_GRAD(bfloat16);

#undef REGISTER_MUSA_GELU_GRAD

}  // namespace musa

REGISTER_OP("MusaGelu")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("approximate: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

REGISTER_OP("MusaGeluGrad")
    .Input("x: T")
    .Input("dy: T")
    .Output("dx: T")
    .Attr("T: {float, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

}  // namespace tensorflow
