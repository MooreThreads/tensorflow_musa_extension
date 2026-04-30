#include <mudnn.h>
#include <mudnn_xmma.h>

#include <cstdlib>

#include "../utils_op.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool ResolveTF32Enabled() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  if (tf32_env == nullptr) {
    return true;
  }
  return std::atoi(tf32_env) != 0;
}

}  // namespace

// The fused op for MusaLinearActivation, which computes
// MatMul + BiasAdd + Activation.
// MatMul + BiasAdd + Relu is executed by mudnn BatchMatMul RunLt with a
// RELU_BIAS epilogue.

template <typename T>
class MusaLinearActivationOp : public MusaOpKernel {
 public:
  explicit MusaLinearActivationOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation", &activation_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    OP_REQUIRES(ctx, activation_ == "relu",
                errors::InvalidArgument(
                    "Unsupported activation for MusaLinearActivation: ",
                    activation_));
    static const bool tf32_enabled_global = ResolveTF32Enabled();
    tf32_enabled_ = tf32_enabled_global;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& bias_input = ctx->input(2);

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " vs ",
                    in1.shape().DebugString()));

    const int64 d0 = in0.dim_size(in0.dims() - 2);
    const int64 d1 = in0.dim_size(in0.dims() - 1);
    const int64 d2 = in1.dim_size(in1.dims() - 2);
    const int64 d3 = in1.dim_size(in1.dims() - 1);

    const int64 m = trans_a_ ? d1 : d0;
    const int64 k = trans_a_ ? d0 : d1;
    const int64 n = trans_b_ ? d2 : d3;
    const int64 k_check = trans_b_ ? d3 : d2;

    OP_REQUIRES(ctx, k == k_check,
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0] mismatch In[1]"));

    TensorShape output_shape = bcast.output_batch_shape();
    output_shape.AddDim(m);
    output_shape.AddDim(n);

    const int channel_dim = output_shape.dims() - 1;
    OP_REQUIRES(
        ctx,
        bias_input.dims() == 1 &&
            bias_input.dim_size(0) == output_shape.dim_size(channel_dim),
        errors::InvalidArgument(
            "Dimension mismatch in BiasAdd of LinearRelu. bias=",
            bias_input.shape().DebugString(),
            ", matmul_out=", output_shape.DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);

    ::musa::dnn::Status status;

    mTensor mt_bias = CreateMTensor(bias_input);

    mBatchMatMul op;
    const auto compute_mode =
        (!tf32_enabled_ && (in0.dtype() == DT_FLOAT || in0.dtype() == DT_DOUBLE))
            ? mBatchMatMul::ComputeMode::SCALAR
            : mBatchMatMul::ComputeMode::TENSOR;
    status = op.SetComputeMode(compute_mode);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN BatchMatMul SetComputeMode failed in LinearActivation."));
    status = op.SetTranspose(trans_a_, trans_b_);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN BatchMatMul SetTranspose failed in LinearActivation."));
    status = op.SetAlpha(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN BatchMatMul SetAlpha failed in LinearActivation."));
    status = op.SetBeta(0.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN BatchMatMul SetBeta failed in LinearActivation."));
    status = op.SetGamma(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN BatchMatMul SetGamma failed in LinearActivation."));

    const int64_t out_batch = bcast.output_batch_shape().num_elements();

    auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
      const int64_t dims = t.dims();
      const int64_t rows = t.dim_size(dims - 2);
      const int64_t cols = t.dim_size(dims - 1);
      const int64_t batch = t.NumElements() / (rows * cols);
      if (dims != 3 || (batch == 1 && out_batch > 1)) {
        mt.SetNdInfo(
            {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
            {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
      }
    };
    ReshapeTo3D(mt_a, in0);
    ReshapeTo3D(mt_b, in1);
    mt_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});

    if (!tf32_enabled_) {
      status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal(
                      "MUSA BatchMatMul bias add execution failed in "
                      "LinearActivation."));

      mUnary relu_op;
      status = relu_op.SetMode(::musa::dnn::Unary::Mode::RELU);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal(
                      "muDNN Unary SetMode(RELU) failed in LinearActivation."));
      status = relu_op.Run(handle, mt_out, mt_out);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal(
                      "MUSA Relu execution failed in LinearActivation."));
      return;
    }

    ::musa::dnn::MatMulLtParam param;
    status = param.SetEpilogue(
        ::musa::dnn::MatMulLtParam::MatMulLtEpilogueMode::
            MATMULLT_EPILOGUE_RELU_BIAS);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "muDNN MatMulLtParam SetEpilogue failed in LinearActivation."));

    status = op.RunLt(handle, mt_out, mt_a, mt_b, mt_out, mt_bias, param);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA BatchMatMul epilogue relu execution failed in "
                    "LinearActivation."));
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;  // TF32 acceleration enabled by default
  std::string activation_ = "relu";
  float alpha_ = 0.0f;
};

#define REGISTER_MUSA_LINEAR_ACTIVATION(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MusaLinearActivation").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLinearActivationOp<TYPE>);

REGISTER_MUSA_LINEAR_ACTIVATION(float);
REGISTER_MUSA_LINEAR_ACTIVATION(Eigen::half);
REGISTER_MUSA_LINEAR_ACTIVATION(bfloat16);
REGISTER_MUSA_LINEAR_ACTIVATION(double);

#undef REGISTER_MUSA_LINEAR_ACTIVATION
}  // namespace musa

REGISTER_OP("MusaLinearActivation")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16, double}")
    .Attr("activation: string = 'relu'")
    .Attr("alpha: float = 0.0")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

}  // namespace tensorflow
