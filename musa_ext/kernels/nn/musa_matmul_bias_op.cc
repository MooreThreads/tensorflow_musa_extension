#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMatMulBiasAddOp : public MusaOpKernel {
 public:
  explicit MusaMatMulBiasAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 2,
                errors::InvalidArgument(
                    "MatMulBiasAdd requires input a to be 2D, got shape ",
                    a.shape().DebugString()));
    OP_REQUIRES(ctx, b.dims() == 2,
                errors::InvalidArgument(
                    "MatMulBiasAdd requires input b to be 2D, got shape ",
                    b.shape().DebugString()));
    OP_REQUIRES(ctx, bias.dims() == 1,
                errors::InvalidArgument(
                    "MatMulBiasAdd requires bias to be 1D, got shape ",
                    bias.shape().DebugString()));

    if (a.NumElements() == 0 || b.NumElements() == 0 ||
        bias.NumElements() == 0) {
      TensorShape out_shape;
      OP_REQUIRES_OK(ctx, ComputeOutputShape(a, b, &out_shape));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
      return;
    }

    const int64_t a_rows = a.dim_size(0);
    const int64_t a_cols = a.dim_size(1);
    const int64_t b_rows = b.dim_size(0);
    const int64_t b_cols = b.dim_size(1);

    const int64_t m = transpose_a_ ? a_cols : a_rows;
    const int64_t k_a = transpose_a_ ? a_rows : a_cols;
    const int64_t k_b = transpose_b_ ? b_cols : b_rows;
    const int64_t n = transpose_b_ ? b_rows : b_cols;

    OP_REQUIRES(ctx, k_a == k_b,
                errors::InvalidArgument("Matrix size-incompatible: a shape ",
                                        a.shape().DebugString(), ", b shape ",
                                        b.shape().DebugString(),
                                        ", transpose_a=", transpose_a_,
                                        ", transpose_b=", transpose_b_));

    OP_REQUIRES(ctx, bias.dim_size(0) == n,
                errors::InvalidArgument("Bias dimension mismatch: bias shape ",
                                        bias.shape().DebugString(),
                                        ", expected [", n, "]"));

    TensorShape out_shape({m, n});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_a = CreateMTensor(a, format_);
    mTensor mt_b = CreateMTensor(b, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    ::musa::dnn::MatMul op;
    auto status = op.SetTranspose(transpose_a_, transpose_b_);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetTranspose failed, status=",
                                 static_cast<int>(status)));

    status = op.SetAlpha(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetAlpha failed, status=",
                                 static_cast<int>(status)));

    status = op.SetBeta(0.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetBeta failed, status=",
                                 static_cast<int>(status)));

    status = op.SetGamma(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetGamma failed, status=",
                                 static_cast<int>(status)));

    status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMulBiasAdd failed, status=",
                                 static_cast<int>(status)));
  }

 private:
  Status ComputeOutputShape(const Tensor& a, const Tensor& b,
                            TensorShape* out_shape) {
    const int64_t a_rows = a.dim_size(0);
    const int64_t a_cols = a.dim_size(1);
    const int64_t b_rows = b.dim_size(0);
    const int64_t b_cols = b.dim_size(1);

    const int64_t m = transpose_a_ ? a_cols : a_rows;
    const int64_t k_a = transpose_a_ ? a_rows : a_cols;
    const int64_t k_b = transpose_b_ ? b_cols : b_rows;
    const int64_t n = transpose_b_ ? b_rows : b_cols;

    if (k_a != k_b) {
      return errors::InvalidArgument(
          "Matrix size-incompatible: a shape ", a.shape().DebugString(),
          ", b shape ", b.shape().DebugString(), ", transpose_a=", transpose_a_,
          ", transpose_b=", transpose_b_);
    }

    *out_shape = TensorShape({m, n});
    return Status::OK();
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

#define REGISTER_MUSA_MATMUL_BIASADD(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MusaMatMulBiasAdd").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMatMulBiasAddOp<TYPE>);

REGISTER_MUSA_MATMUL_BIASADD(float);
// REGISTER_MUSA_MATMUL_BIASADD(double);
REGISTER_MUSA_MATMUL_BIASADD(Eigen::half);
REGISTER_MUSA_MATMUL_BIASADD(bfloat16);

#undef REGISTER_MUSA_MATMUL_BIASADD

}  // namespace musa

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("MusaMatMulBiasAdd")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);
}  // namespace tensorflow
