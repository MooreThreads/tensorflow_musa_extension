#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

namespace {

extern "C" void LaunchMusaSmallLastDimSoftmaxFloat(const float* input,
                                                    float* output,
                                                    int64_t outer_size,
                                                    int last_dim,
                                                    musaStream_t stream);

extern "C" void LaunchMusaSmallLastDimCausalMaskedSoftmaxFloat(
    const float* input, const float* scale, float* output, int64_t outer_size,
    int query_dim, int key_dim, musaStream_t stream);
extern "C" void LaunchMusaSmallLastDimCausalMaskedSoftmaxBFloat16(
    const tensorflow::bfloat16* input, const tensorflow::bfloat16* scale,
    tensorflow::bfloat16* output, int64_t outer_size, int query_dim,
    int key_dim, musaStream_t stream);

bool EnvFlagValueIsFalse(const char* value) {
  return value != nullptr && value[0] != '\0' &&
         (value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
          value[0] == 'n' || value[0] == 'N' || value[0] == 'o' ||
          value[0] == 'O');
}

bool EnvFlagValueIsTrue(const char* value) {
  return value != nullptr && value[0] != '\0' && !EnvFlagValueIsFalse(value);
}

bool SmallLastDimSoftmaxDisabledByEnv() {
  if (EnvFlagValueIsTrue(std::getenv("MUSA_DISABLE_SMALL_LASTDIM_SOFTMAX"))) {
    return true;
  }

  // Legacy compatibility: existing scripts can still opt out with ENABLE=0.
  return EnvFlagValueIsFalse(std::getenv("MUSA_ENABLE_SMALL_LASTDIM_SOFTMAX"));
}

bool ShouldUseSmallLastDimSoftmax(const Tensor& logits) {
  if (SmallLastDimSoftmaxDisabledByEnv()) return false;
  if (logits.dtype() != DT_FLOAT) return false;
  const int dims = logits.dims();
  if (dims < 2) return false;
  const int64_t last_dim = logits.dim_size(dims - 1);
  if (last_dim <= 0 || last_dim > 64) return false;
  const int64_t outer_size = logits.NumElements() / last_dim;
  return outer_size >= 1024;
}

class MusaSoftmaxCall : public MusaOpKernel {
 public:
  explicit MusaSoftmaxCall(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  // Softmax is computationally intensive (exp + reduction)
  // Mark as expensive to enable optimal scheduling (async execution)
  // Expected improvement: Better overlapping with other operations
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in.shape().DebugString()));

    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in.shape(), &softmax_out));

    if (logits_in.NumElements() == 0) return;

    auto in = CreateMTensor(logits_in, format_);
    auto out = CreateMTensor(*softmax_out, format_);

    MUSA_OP_REQUIRES_MUDNN_HANDLE(context);
    auto& h = GetHandleByCtx(context);
    mSoftmax softmax_op;
    this->Operate(softmax_op);

    int axis = static_cast<int>(logits_in.dims() - 1);
    MTOP_CHECK_OK(softmax_op.SetDim(axis), "SetDim", context);
    MTOP_CHECK_OK(softmax_op.SetAlgorithm(mSoftmax::Algorithm::ACCURATE),
                  "SetAlgorithm", context);

    MTOP_CHECK_OK_RUN(softmax_op.Run(h, out, in), "RunSoftmax", context);
  }

  virtual void Operate(mSoftmax& op) = 0;
};

template <SOFTMAX_MODE m>
class MusaSoftmaxOp : public MusaSoftmaxCall {
 public:
  using MusaSoftmaxCall::MusaSoftmaxCall;
  void Operate(mSoftmax& op) override { op.SetMode(m); }
};

class MusaCausalMaskedSoftmaxOp : public MusaOpKernel {
 public:
  explicit MusaCausalMaskedSoftmaxOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    OP_REQUIRES(ctx, logits.dtype() == DT_FLOAT || logits.dtype() == DT_BFLOAT16,
                errors::InvalidArgument(
                    "MusaCausalMaskedSoftmax expects float/bfloat16 logits"));
    OP_REQUIRES(ctx, scale.dtype() == logits.dtype(),
                errors::InvalidArgument(
                    "MusaCausalMaskedSoftmax expects scale dtype to match logits"));
    OP_REQUIRES(ctx, scale.NumElements() == 1,
                errors::InvalidArgument(
                    "MusaCausalMaskedSoftmax expects scalar scale, got ",
                    scale.shape().DebugString()));
    OP_REQUIRES(ctx, logits.dims() == 4,
                errors::InvalidArgument(
                    "MusaCausalMaskedSoftmax expects rank-4 logits, got ",
                    logits.shape().DebugString()));

    const int64_t query_dim = logits.dim_size(2);
    const int64_t key_dim = logits.dim_size(3);
    OP_REQUIRES(ctx, query_dim > 0 && key_dim > 0 && query_dim <= key_dim &&
                         key_dim <= 64,
                errors::InvalidArgument(
                    "MusaCausalMaskedSoftmax invalid attention shape: ",
                    logits.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, logits.shape(), &output));
    if (logits.NumElements() == 0) return;

    const int64_t outer_size = logits.NumElements() / key_dim;
    auto& h = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(h.GetStream());
    if (logits.dtype() == DT_FLOAT) {
      LaunchMusaSmallLastDimCausalMaskedSoftmaxFloat(
          logits.flat<float>().data(), scale.flat<float>().data(),
          output->flat<float>().data(), outer_size, static_cast<int>(query_dim),
          static_cast<int>(key_dim), stream);
    } else {
      LaunchMusaSmallLastDimCausalMaskedSoftmaxBFloat16(
          logits.flat<bfloat16>().data(), scale.flat<bfloat16>().data(),
          output->flat<bfloat16>().data(), outer_size,
          static_cast<int>(query_dim), static_cast<int>(key_dim), stream);
    }
  }
};

}  // namespace

#define REGISTER_MUSA_SOFTMAX_TYPE(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Softmax").Device("MUSA").TypeConstraint<TYPE>("T"),    \
      MusaSoftmaxOp<SOFTMAX_MODE::SOFTMAX>);                       \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("LogSoftmax").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSoftmaxOp<SOFTMAX_MODE::LOGSOFTMAX>);

REGISTER_MUSA_SOFTMAX_TYPE(float);
REGISTER_MUSA_SOFTMAX_TYPE(Eigen::half);
REGISTER_MUSA_SOFTMAX_TYPE(bfloat16);
REGISTER_MUSA_SOFTMAX_TYPE(double);
REGISTER_MUSA_SOFTMAX_TYPE(int32);
REGISTER_MUSA_SOFTMAX_TYPE(int64);

#undef REGISTER_MUSA_SOFTMAX_TYPE

REGISTER_KERNEL_BUILDER(Name("MusaCausalMaskedSoftmax").Device("MUSA"),
                        MusaCausalMaskedSoftmaxOp);

}  // namespace musa

REGISTER_OP("MusaCausalMaskedSoftmax")
    .Input("logits: T")
    .Input("scale: T")
    .Output("softmax: T")
    .Attr("T: {float, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

}  // namespace tensorflow
