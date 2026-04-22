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
    MUSA_KERNEL_TIMING_GUARD(ctx);
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
    OP_REQUIRES(ctx,
                bias_input.dims() == 1 &&
                    bias_input.dim_size(0) == output_shape.dim_size(channel_dim),
                errors::InvalidArgument(
                    "Dimension mismatch in BiasAdd of LinearRelu. bias=",
                    bias_input.shape().DebugString(), ", matmul_out=",
                    output_shape.DebugString()));

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

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      status = op.Run(handle, mt_out, mt_a, mt_b);
    } else {
      mBatchMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
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
      status = op.Run(handle, mt_out, mt_a, mt_b);
    }

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA MatMul/BatchMatMul execution failed in LinearRelu."));

    const T* bias_ptr = bias_input.flat<T>().data();
    T* out_ptr = output->flat<T>().data();
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    MUSA_KERNEL_TRACE_START("BiasAddReluKernel");
    LaunchBiasAddReluKernel(out_ptr, bias_ptr, out_ptr, output->NumElements(),
                            output_shape.dim_size(channel_dim), stream);
    MUSA_KERNEL_TRACE_END("BiasAddReluKernel");

    const musaError_t launch_status = musaGetLastError();
    OP_REQUIRES(ctx, launch_status == musaSuccess,
                errors::Internal("MUSA BiasAddRelu kernel launch failed in "
                                 "LinearRelu: ",
                                 musaGetErrorString(launch_status)));
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;
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
