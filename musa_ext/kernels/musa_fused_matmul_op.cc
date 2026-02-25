#include <mudnn.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils_op.h"

#define ENABLE_MUSA_DEBUG 0

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaFusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, double, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("fused_ops: list(string) = []")
    .Attr("num_args: int >= 0 = 0")
    .Attr("epsilon: float = 0.0001")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

template <typename T>
class MusaFusedMatMulOp : public MusaOpKernel {
 public:
  explicit MusaFusedMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    if (ctx->HasAttr("transpose_a")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_x_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_y_));
    } else {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &trans_x_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &trans_y_));
    }

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));

    if (fused_ops.size() == 1 && fused_ops[0] == "BiasAdd") {
      fusion_type_ = FusionType::BIAS_ADD;
      fusion_name_ = "BiasAdd";
    } else if (fused_ops.size() == 2 && fused_ops[0] == "BiasAdd" &&
               fused_ops[1] == "Relu") {
      fusion_type_ = FusionType::BIAS_ADD_RELU;
      fusion_name_ = "BiasAdd+Relu";
    } else {
      fusion_type_ = FusionType::BIAS_ADD;
      fusion_name_ = "Unknown(Default=BiasAdd)";
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes"));

    auto dims0 = in0.dims();
    auto dims1 = in1.dims();

    if (dims0 < 2 || dims1 < 2) {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Input tensors must have rank >= 2"));
      return;
    }

    int64 m = in0.dim_size(dims0 - 2);
    int64 k = in0.dim_size(dims0 - 1);
    if (trans_x_) std::swap(m, k);

    int64 n = in1.dim_size(dims1 - 1);
    if (trans_y_) n = in1.dim_size(dims1 - 2);

    TensorShape out_shape = bcast.output_batch_shape();
    out_shape.AddDim(m);
    out_shape.AddDim(n);

    OP_REQUIRES(ctx, ctx->num_inputs() >= 3,
                errors::InvalidArgument("FusedMatMul requires Bias input"));
    const Tensor& bias = ctx->input(2);
    OP_REQUIRES(ctx, bias.dims() == 1,
                errors::InvalidArgument("Bias must be 1D"));
    OP_REQUIRES(ctx, bias.dim_size(0) == n,
                errors::InvalidArgument("Bias dim mismatch"));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    {
      mBatchMatMul matmul_op;
      matmul_op.SetTranspose(trans_x_, trans_y_);
      matmul_op.SetAlpha(1.0);
      matmul_op.SetBeta(0.0);

      mTensor mt0 = CreateMTensor(in0, format_);
      mTensor mt1 = CreateMTensor(in1, format_);
      mTensor mt_out = CreateMTensor(*out, format_);

      std::vector<std::vector<int64_t>> shape_storage;
      shape_storage.reserve(6);

      auto FixToBatchFormat = [&](mTensor& mt, const Tensor& t) {
        if (t.dims() == 2) {
          int64_t rows = t.dim_size(0);
          int64_t cols = t.dim_size(1);

          std::vector<int64_t> dims = {1, rows, cols};
          std::vector<int64_t> strides = {rows * cols, cols, 1};

          shape_storage.push_back(dims);
          shape_storage.push_back(strides);

          int64_t* p_dims = shape_storage[shape_storage.size() - 2].data();
          int64_t* p_strides = shape_storage[shape_storage.size() - 1].data();

          mt.SetNdInfo(3, p_dims, p_strides);
        }
      };

      FixToBatchFormat(mt0, in0);
      FixToBatchFormat(mt1, in1);
      FixToBatchFormat(mt_out, *out);

      auto status = matmul_op.Run(handle, mt_out, mt0, mt1);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MatMul failed. Status: ", (int)status));
    }

    {
      mBinary binary_op;
      binary_op.SetMode(::musa::dnn::Binary::Mode::ADD);

      mTensor mt_out = CreateMTensor(*out, format_);
      mTensor mt_bias = CreateMTensor(bias, format_);

      auto status = binary_op.Run(handle, mt_out, mt_out, mt_bias);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("BiasAdd failed. Status: ", (int)status));
    }

    if (fusion_type_ == FusionType::BIAS_ADD_RELU) {
      mTensor mt_out = CreateMTensor(*out, format_);

      mUnary unary_op;
      unary_op.SetMode(::musa::dnn::Unary::Mode::RELU);

      auto status = unary_op.Run(handle, mt_out, mt_out);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("Fused ReLU failed. Status: ", (int)status));
    }
  }

 private:
  bool trans_x_ = false;
  bool trans_y_ = false;
  std::string fusion_name_;
  enum class FusionType { BIAS_ADD, BIAS_ADD_RELU };
  FusionType fusion_type_;
};

#define REGISTER_MUSA_FUSED_MATMUL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaFusedMatMul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFusedMatMulOp<TYPE>);

REGISTER_MUSA_FUSED_MATMUL(float);
REGISTER_MUSA_FUSED_MATMUL(double);
REGISTER_MUSA_FUSED_MATMUL(Eigen::half);
REGISTER_MUSA_FUSED_MATMUL(bfloat16);

#undef REGISTER_MUSA_FUSED_MATMUL

}  // namespace musa
}  // namespace tensorflow