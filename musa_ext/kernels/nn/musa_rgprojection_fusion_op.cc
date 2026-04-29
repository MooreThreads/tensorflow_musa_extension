#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// The fused op for MusaBiasAddReluMatMul, which computes:
//   if relu_input_slot == 0:
//     MatMul(Relu(BiasAdd(x, bias)), other)
//   else:
//     MatMul(other, Relu(BiasAdd(x, bias)))
//
// Provides a mudnn-based implementation:
// 1) BiasAdd + Relu on the first input
// 2) MatMul with the other input

template <typename T>
class MusaBiasAddReluMatMulOp : public MusaOpKernel {
 public:
  explicit MusaBiasAddReluMatMulOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu_input_slot", &relu_input_slot_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));

    OP_REQUIRES(ctx, relu_input_slot_ == 0 || relu_input_slot_ == 1,
                errors::InvalidArgument("relu_input_slot must be 0 or 1, got ",
                                        relu_input_slot_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& bias_input = ctx->input(1);
    const Tensor& other = ctx->input(2);

    OP_REQUIRES(ctx, input.dims() >= 2,
                errors::InvalidArgument("input must have rank >= 2, got ",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, other.dims() >= 2,
                errors::InvalidArgument("other must have rank >= 2, got ",
                                        other.shape().DebugString()));
    OP_REQUIRES(ctx, bias_input.dims() == 1,
                errors::InvalidArgument("bias must be rank 1, got ",
                                        bias_input.shape().DebugString()));

    // 1. BiasAdd + Relu on input
    TensorShape bias_relu_shape = input.shape();
    OP_REQUIRES(
        ctx, bias_input.dim_size(0) == input.dim_size(input.dims() - 1),
        errors::InvalidArgument(
            "Dimension mismatch in BiasAddReluMatMul BiasAdd: input shape=",
            input.shape().DebugString(),
            ", bias shape=", bias_input.shape().DebugString(),
            ", expected bias dim == input last dim"));

    Tensor bias_relu_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), bias_relu_shape,
                                           &bias_relu_tensor));

    if (bias_relu_tensor.NumElements() == 0) {
      Tensor* output = nullptr;
      TensorShape out_shape;
      OP_REQUIRES_OK(ctx, InferOutputShape(input, other, &out_shape));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
      return;
    }

    RunBiasAddRelu(ctx, input, bias_input, bias_relu_tensor);

    // 2. MatMul
    const Tensor* lhs = nullptr;
    const Tensor* rhs = nullptr;

    if (relu_input_slot_ == 0) {
      lhs = &bias_relu_tensor;
      rhs = &other;
    } else {
      lhs = &other;
      rhs = &bias_relu_tensor;
    }

    TensorShape mm_out_shape;
    OP_REQUIRES_OK(ctx, InferMatMulOutputShape(*lhs, *rhs, &mm_out_shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

    RunMatMul(ctx, *lhs, *rhs, *output);
  }

  bool IsExpensive() override { return true; }

 private:
  int relu_input_slot_ = 0;
  bool trans_a_ = false;
  bool trans_b_ = false;
  bool tf32_enabled_ = false;

  Status InferOutputShape(const Tensor& input, const Tensor& other,
                          TensorShape* out_shape) {
    const Tensor* lhs = nullptr;
    const Tensor* rhs = nullptr;

    if (relu_input_slot_ == 0) {
      lhs = &input;
      rhs = &other;
    } else {
      lhs = &other;
      rhs = &input;
    }

    return InferMatMulOutputShape(*lhs, *rhs, out_shape);
  }

  Status InferMatMulOutputShape(const Tensor& lhs, const Tensor& rhs,
                                TensorShape* out_shape) {
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    if (!bcast.IsValid()) {
      return errors::InvalidArgument(
          "Incompatible shapes for MatMulBCast: ", lhs.shape().DebugString(),
          " vs ", rhs.shape().DebugString());
    }

    int64 d0 = lhs.dim_size(lhs.dims() - 2);
    int64 d1 = lhs.dim_size(lhs.dims() - 1);
    int64 d2 = rhs.dim_size(rhs.dims() - 2);
    int64 d3 = rhs.dim_size(rhs.dims() - 1);

    int64 m = trans_a_ ? d1 : d0;
    int64 k = trans_a_ ? d0 : d1;
    int64 n = trans_b_ ? d2 : d3;
    int64 k_check = trans_b_ ? d3 : d2;

    if (k != k_check) {
      return errors::InvalidArgument(
          "Matrix size-incompatible in BiasAddReluMatMul: lhs=",
          lhs.shape().DebugString(), ", rhs=", rhs.shape().DebugString(),
          ", transpose_a=", trans_a_, ", transpose_b=", trans_b_);
    }

    *out_shape = bcast.output_batch_shape();
    out_shape->AddDim(m);
    out_shape->AddDim(n);
    return Status::OK();
  }

  void RunBiasAddRelu(OpKernelContext* ctx, const Tensor& input,
                      const Tensor& bias_input, Tensor& bias_relu_tensor) {
    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_input = CreateMTensor(input);
    mTensor mt_bias = CreateMTensor(bias_input);
    mTensor mt_tmp = CreateMTensor(bias_relu_tensor);

    int channel_dim = input.dims() - 1;
    int dims_cnt = input.dims();

    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias_input.dim_size(0);
    b_strides[channel_dim] = 1;
    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    mBinary bias_op;
    bias_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    mStatus status = bias_op.Run(handle, mt_tmp, mt_input, mt_bias);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA BiasAdd failed in BiasAddReluMatMul."));

    mUnary relu_op;
    relu_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    status = relu_op.Run(handle, mt_tmp, mt_tmp);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Relu failed in BiasAddReluMatMul."));
  }

  void RunMatMul(OpKernelContext* ctx, const Tensor& lhs, const Tensor& rhs,
                 Tensor& output) {
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", lhs.shape().DebugString(), " vs ",
                    rhs.shape().DebugString()));

    int64 d0 = lhs.dim_size(lhs.dims() - 2);
    int64 d1 = lhs.dim_size(lhs.dims() - 1);
    int64 d2 = rhs.dim_size(rhs.dims() - 2);
    int64 d3 = rhs.dim_size(rhs.dims() - 1);

    int64 m = trans_a_ ? d1 : d0;
    int64 k = trans_a_ ? d0 : d1;
    int64 n = trans_b_ ? d2 : d3;
    int64 k_check = trans_b_ ? d3 : d2;

    OP_REQUIRES(
        ctx, k == k_check,
        errors::InvalidArgument("Matrix size-incompatible: lhs mismatch rhs"));

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    mTensor mt_a = CreateMTensor(lhs);
    mTensor mt_b = CreateMTensor(rhs);
    mTensor mt_out = CreateMTensor(output);

    ::musa::dnn::Status status;

    if (lhs.dims() == 2 && rhs.dims() == 2) {
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

      ReshapeTo3D(mt_a, lhs);
      ReshapeTo3D(mt_b, rhs);
      mt_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});
      status = op.Run(handle, mt_out, mt_a, mt_b);
    }

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal(
            "MUSA MatMul/BatchMatMul execution failed in BiasAddReluMatMul."));
  }
};

#define REGISTER_MUSA_BIASADD_RELU_MATMUL(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("MusaBiasAddReluMatMul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaBiasAddReluMatMulOp<TYPE>);

REGISTER_MUSA_BIASADD_RELU_MATMUL(float);
REGISTER_MUSA_BIASADD_RELU_MATMUL(Eigen::half);
REGISTER_MUSA_BIASADD_RELU_MATMUL(bfloat16);
// REGISTER_MUSA_BIASADD_RELU_MATMUL(double);

#undef REGISTER_MUSA_BIASADD_RELU_MATMUL

}  // namespace musa

REGISTER_OP("MusaBiasAddReluMatMul")
    .Input("input: T")
    .Input("bias: T")
    .Input("other: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("relu_input_slot: int")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using ::tensorflow::shape_inference::DimensionHandle;
      using ::tensorflow::shape_inference::ShapeHandle;

      ShapeHandle input;
      ShapeHandle bias;
      ShapeHandle other;

      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &bias));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &other));

      int relu_input_slot;
      bool transpose_a;
      bool transpose_b;
      TF_RETURN_IF_ERROR(c->GetAttr("relu_input_slot", &relu_input_slot));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));

      DimensionHandle merged_bias_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(bias, 0), &merged_bias_dim));

      ShapeHandle lhs;
      ShapeHandle rhs;
      if (relu_input_slot == 0) {
        lhs = input;
        rhs = other;
      } else if (relu_input_slot == 1) {
        lhs = other;
        rhs = input;
      } else {
        return errors::InvalidArgument("relu_input_slot must be 0 or 1");
      }

      DimensionHandle lhs_rows = c->Dim(lhs, transpose_a ? -1 : -2);
      DimensionHandle lhs_cols = c->Dim(lhs, transpose_a ? -2 : -1);
      DimensionHandle rhs_rows = c->Dim(rhs, transpose_b ? -1 : -2);
      DimensionHandle rhs_cols = c->Dim(rhs, transpose_b ? -2 : -1);

      DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(lhs_cols, rhs_rows, &merged));

      ShapeHandle lhs_batch;
      ShapeHandle rhs_batch;
      ShapeHandle batch_out;
      TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch));
      TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch));
      TF_RETURN_IF_ERROR(c->Merge(lhs_batch, rhs_batch, &batch_out));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(batch_out, c->Vector(lhs_rows), &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(rhs_cols), &out));
      c->set_output(0, out);
      return Status::OK();
    });

}  // namespace tensorflow
