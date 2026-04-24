#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// The fused op for MusaLinearRelu, which computes MatMul + BiasAdd + Relu
// Provides two types of implementations:
// 1) A pure MUSA implementation using mudnn for MatMul and a custom kernel for
// BiasAdd+Relu
// 2) A fallback implementation that uses mudnn for MatMul and then a separate
// kernel for BiasAdd+Relu (for better performance on smaller sizes)

template <typename T>
void LaunchBiasAddReluKernel(const T*, const T*, T*, int, int, musaStream_t);

template <typename T>
class MusaLinearReluOp : public MusaOpKernel {
 public:
  explicit MusaLinearReluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& bias_input = ctx->input(2);

    // 1. MatMul
    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " vs ",
                    in1.shape().DebugString()));

    int64 d0 = in0.dim_size(in0.dims() - 2);
    int64 d1 = in0.dim_size(in0.dims() - 1);
    int64 d2 = in1.dim_size(in1.dims() - 2);
    int64 d3 = in1.dim_size(in1.dims() - 1);

    int64 m = trans_a_ ? d1 : d0;
    int64 k = trans_a_ ? d0 : d1;
    int64 n = trans_b_ ? d2 : d3;
    int64 k_check = trans_b_ ? d3 : d2;

    OP_REQUIRES(ctx, k == k_check,
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0] mismatch In[1]"));

    TensorShape mm_out_shape = bcast.output_batch_shape();
    mm_out_shape.AddDim(m);
    mm_out_shape.AddDim(n);

    Tensor mm_out_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(in0.dtype(), mm_out_shape, &mm_out_tensor));

    if (mm_out_tensor.NumElements() == 0) {
      Tensor* final_output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &final_output));
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_mm_out = CreateMTensor(mm_out_tensor);

    ::musa::dnn::Status status;

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      status = op.Run(handle, mt_mm_out, mt_a, mt_b);
    } else {
      mBatchMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      int64_t out_batch = bcast.output_batch_shape().num_elements();

      auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
        int64_t dims = t.dims();
        int64_t rows = t.dim_size(dims - 2);
        int64_t cols = t.dim_size(dims - 1);
        int64_t batch = t.NumElements() / (rows * cols);
        if (dims != 3 || (batch == 1 && out_batch > 1)) {
          mt.SetNdInfo(
              {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
              {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
        }
      };
      ReshapeTo3D(mt_a, in0);
      ReshapeTo3D(mt_b, in1);
      mt_mm_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});
      status = op.Run(handle, mt_mm_out, mt_a, mt_b);
    }

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA MatMul/BatchMatMul execution failed in LinearRelu."));

    // 2. BiasAdd + Relu
    UseMudnn(ctx, bias_input, mm_out_shape, mt_mm_out);
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;  // TF32 acceleration enabled by default

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
