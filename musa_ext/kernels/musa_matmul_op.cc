#include <mudnn.h>
#include <mudnn_xmma.h>

#include <cstdlib>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

REGISTER_OP("MusaMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(shape_inference::MatMulShape);

template <typename T>
class MusaMatMulOp : public MusaOpKernel {
 public:
  explicit MusaMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    trans_a_ = false;
    trans_b_ = false;

    if (ctx->HasAttr("transpose_a")) ctx->GetAttr("transpose_a", &trans_a_);
    if (ctx->HasAttr("transpose_b")) ctx->GetAttr("transpose_b", &trans_b_);

    bool adj_x = false;
    bool adj_y = false;
    if (ctx->GetAttr("adj_x", &adj_x).ok()) trans_a_ = adj_x;
    if (ctx->GetAttr("adj_y", &adj_y).ok()) trans_b_ = adj_y;

    // TF32 is disabled by default for better precision
    // Can be enabled via MUSA_ENABLE_TF32=1 environment variable
    // Use static initialization to cache environment variable lookup
    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      return (tf32_env && std::atoi(tf32_env) == 1);
    }();
    tf32_enabled_ = tf32_enabled_global;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

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

    TensorShape out_shape = bcast.output_batch_shape();
    out_shape.AddDim(m);
    out_shape.AddDim(n);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);  // Use TF32 setting from constructor
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*out);

    auto FixToBatchFormat = [](mTensor& mt, const Tensor& t) {
      if (t.dims() == 2) {
        int64_t rows = t.dim_size(0);
        int64_t cols = t.dim_size(1);
        mt.SetNdInfo({1, rows, cols}, {rows * cols, cols, 1});
      }
    };

    ::musa::dnn::Status status;

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);

      status = op.Run(handle, mt_out, mt_a, mt_b);

      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal(
              "MUSA MatMul (2D High Precision) execution failed. Status: ",
              (int)status));
    } else {
      mBatchMatMul op;
      op.SetTranspose(trans_a_, trans_b_);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);

      FixToBatchFormat(mt_a, in0);
      FixToBatchFormat(mt_b, in1);
      FixToBatchFormat(mt_out, *out);

      status = op.Run(handle, mt_out, mt_a, mt_b);

      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal("MUSA BatchMatMul execution failed. Status: ",
                           (int)status));
    }
  }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;  // TF32 acceleration enabled by default
};

#define REGISTER_MUSA_MATMUL_ALL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatMul").Device("MUSA").TypeConstraint<TYPE>("T"),            \
      MusaMatMulOp<TYPE>);                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device("MUSA").TypeConstraint<TYPE>("T"),     \
      MusaMatMulOp<TYPE>);                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MusaMatMul").Device("MUSA").TypeConstraint<TYPE>("T"),        \
      MusaMatMulOp<TYPE>);                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MusaBatchMatMulV2").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMatMulOp<TYPE>);

REGISTER_MUSA_MATMUL_ALL(float);
REGISTER_MUSA_MATMUL_ALL(double);
REGISTER_MUSA_MATMUL_ALL(Eigen::half);
REGISTER_MUSA_MATMUL_ALL(bfloat16);

#undef REGISTER_MUSA_MATMUL_ALL

}  // namespace musa
}  // namespace tensorflow
