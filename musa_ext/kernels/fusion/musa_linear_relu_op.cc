#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

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

// The fused op for MusaLinearRelu, which computes MatMul + BiasAdd + Relu.
// Write MatMul directly into the final output and then apply the fused epilogue
// in-place to avoid an additional large temporary tensor and two extra mudnn
// launches.

template <typename T>
void LaunchBiasAddReluKernel(const T* x, const T* bias, T* output,
                             int64_t n_elements, int64_t n_cols,
                             musaStream_t stream);

template <typename T>
class MusaLinearReluOp : public MusaOpKernel {
 public:
  explicit MusaLinearReluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));

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

    TensorShape out_shape = bcast.output_batch_shape();
    out_shape.AddDim(m);
    out_shape.AddDim(n);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);
    mTensor mt_bias = CreateMTensor(bias_input);

    ::musa::dnn::Status status;

    // Fused MatMul+BiasAdd via RunWithBiasAdd, then Relu in-place
    // Saves 1 kernel launch vs separate MatMul + BiasAdd + Relu
    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      op.SetGamma(1.0);
      status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);
    } else {
      mBatchMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      op.SetGamma(1.0);
      int64_t out_batch = bcast.output_batch_shape().num_elements();

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
      status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);
    }

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA RunWithBiasAdd failed in LinearRelu."));

    // Relu in-place on output
    mUnary relu_op;
    relu_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    status = relu_op.Run(handle, mt_out, mt_out);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Relu failed in LinearRelu."));
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = true;  // TF32 acceleration enabled by default

  void UseMudnn(OpKernelContext* ctx, const Tensor& bias_input,
                const TensorShape& mm_out_shape, const mTensor& mt_mm_out) {
    auto& handle = GetHandleByCtx(ctx);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));

    mTensor mt_bias = CreateMTensor(bias_input);
    mTensor mt_out = CreateMTensor(*output);

    int channel_dim = mm_out_shape.dims() - 1;
    OP_REQUIRES(
        ctx, bias_input.dim_size(0) == mm_out_shape.dim_size(channel_dim),
        errors::InvalidArgument("Dimension mismatch in BiasAdd of LinearRelu"));

    int dims_cnt = mm_out_shape.dims();
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias_input.dim_size(0);
    b_strides[channel_dim] = 1;

    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    mBinary bias_op;
    bias_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    mStatus status = bias_op.Run(handle, mt_out, mt_mm_out, mt_bias);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA BiasAdd failed in LinearRelu."));

    // 3. Relu (In-place on current output)
    mUnary relu_op;
    relu_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    status = relu_op.Run(handle, mt_out, mt_out);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Relu failed in LinearRelu."));
  }

  void UseKernel(OpKernelContext* ctx, const Tensor& bias_input,
                 const TensorShape& mm_out_shape, const Tensor& mm_out_tensor) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    const T* mm_ptr = mm_out_tensor.flat<T>().data();
    LaunchBiasAddReluKernel(
        mm_ptr, bias_input.flat<T>().data(), output->flat<T>().data(),
        mm_out_shape.num_elements(),
        mm_out_shape.dim_size(mm_out_shape.dims() - 1), stream);
  }
};

#define REGISTER_MUSA_LINEAR_RELU(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MusaLinearRelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaLinearReluOp<TYPE>);

REGISTER_MUSA_LINEAR_RELU(float);
REGISTER_MUSA_LINEAR_RELU(Eigen::half);
REGISTER_MUSA_LINEAR_RELU(bfloat16);
REGISTER_MUSA_LINEAR_RELU(double);

#undef REGISTER_MUSA_LINEAR_RELU
}  // namespace musa

REGISTER_OP("MusaLinearRelu")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

}  // namespace tensorflow
