#include <mudnn.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {
namespace musa {

namespace {

// Infer the canonical MatMul output shape and last-dimension width. The
// two-layer fused matmul op reuses this for both linear layers so the same
// validation logic is applied to the first and second MatMul branches.

Status InferMatMulOutputShape(const TensorShape& lhs_shape,
                              const TensorShape& rhs_shape, bool trans_a,
                              bool trans_b, TensorShape* out_shape,
                              int64* output_cols) {
  MatMulBCast bcast(lhs_shape.dim_sizes(), rhs_shape.dim_sizes());
  if (!bcast.IsValid()) {
    return errors::InvalidArgument("Incompatible shapes: ",
                                   lhs_shape.DebugString(), " vs ",
                                   rhs_shape.DebugString());
  }

  if (lhs_shape.dims() < 2 || rhs_shape.dims() < 2) {
    return errors::InvalidArgument("Input tensors must have rank >= 2");
  }

  const int64 lhs_dim0 = lhs_shape.dim_size(lhs_shape.dims() - 2);
  const int64 lhs_dim1 = lhs_shape.dim_size(lhs_shape.dims() - 1);
  const int64 rhs_dim0 = rhs_shape.dim_size(rhs_shape.dims() - 2);
  const int64 rhs_dim1 = rhs_shape.dim_size(rhs_shape.dims() - 1);

  const int64 m = trans_a ? lhs_dim1 : lhs_dim0;
  const int64 k = trans_a ? lhs_dim0 : lhs_dim1;
  const int64 n = trans_b ? rhs_dim0 : rhs_dim1;
  const int64 k_check = trans_b ? rhs_dim1 : rhs_dim0;

  if (k != k_check) {
    return errors::InvalidArgument("Matrix size incompatible: lhs k=", k,
                                   ", rhs k=", k_check, ", lhs shape=",
                                   lhs_shape.DebugString(), ", rhs shape=",
                                   rhs_shape.DebugString());
  }

  *out_shape = bcast.output_batch_shape();
  out_shape->AddDim(m);
  out_shape->AddDim(n);
  *output_cols = n;
  return Status::OK();
}

}  // namespace

// Fused block op for:
//   MatMul + BiasAdd + Relu + MatMul + BiasAdd
//   MatMul + BiasAdd + LeakyRelu + MatMul + BiasAdd
//
// This op is intentionally narrow: it only serves the two-layer linear block
// emitted by MusaMatMulBiasFusion. Internally it still runs two MatMul stages,
// but keeps the intermediate hidden activation inside a single fused op
// boundary and reuses the existing MatMul+BiasAdd epilogue path when possible.
REGISTER_OP("MusaTwoLayerFusedMatMul")
    .Input("a: T")
    .Input("b0: T")
    .Input("bias0: T")
    .Input("other1: T")
    .Input("bias1: T")
    .Output("product: T")
    .Attr("T: {float, half, double, bfloat16}")
    .Attr("transpose_a0: bool = false")
    .Attr("transpose_b0: bool = false")
    .Attr("transpose_a1: bool = false")
    .Attr("transpose_b1: bool = false")
    .Attr("hidden_input_idx1: int = 0")
    .Attr("activation_type: string = 'Relu'")
    .Attr("activation_alpha: float = 0.2")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

template <typename T>
class MusaTwoLayerFusedMatMulOp : public MusaOpKernel {
 public:
  explicit MusaTwoLayerFusedMatMulOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a0", &transpose_a0_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b0", &transpose_b0_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a1", &transpose_a1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b1", &transpose_b1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_input_idx1", &hidden_input_idx1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_type", &activation_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_alpha", &activation_alpha_));
    OP_REQUIRES(ctx, hidden_input_idx1_ == 0 || hidden_input_idx1_ == 1,
                errors::InvalidArgument(
                    "hidden_input_idx1 must be 0 or 1, got ",
                    hidden_input_idx1_));
    OP_REQUIRES(
        ctx, activation_type_ == "Relu" || activation_type_ == "LeakyRelu",
        errors::InvalidArgument("activation_type must be Relu or LeakyRelu, got ",
                                activation_type_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& weight0 = ctx->input(1);
    const Tensor& bias0 = ctx->input(2);
    const Tensor& second_other = ctx->input(3);
    const Tensor& bias1 = ctx->input(4);

    TensorShape hidden_shape;
    int64 hidden_cols = 0;
    OP_REQUIRES_OK(ctx, InferMatMulOutputShape(input.shape(), weight0.shape(),
                                               transpose_a0_, transpose_b0_,
                                               &hidden_shape, &hidden_cols));
    OP_REQUIRES(ctx, bias0.dims() == 1,
                errors::InvalidArgument("bias0 must be 1D"));
    OP_REQUIRES(ctx, bias0.dim_size(0) == hidden_cols,
                errors::InvalidArgument("bias0 dim mismatch: expected ",
                                        hidden_cols, ", got ",
                                        bias0.dim_size(0)));

    TensorShape output_shape;
    int64 output_cols = 0;
    const TensorShape& second_lhs_shape =
        hidden_input_idx1_ == 0 ? hidden_shape : second_other.shape();
    const TensorShape& second_rhs_shape =
        hidden_input_idx1_ == 0 ? second_other.shape() : hidden_shape;
    OP_REQUIRES_OK(ctx, InferMatMulOutputShape(second_lhs_shape,
                                               second_rhs_shape,
                                               transpose_a1_, transpose_b1_,
                                               &output_shape, &output_cols));
    OP_REQUIRES(ctx, bias1.dims() == 1,
                errors::InvalidArgument("bias1 must be 1D"));
    OP_REQUIRES(ctx, bias1.dim_size(0) == output_cols,
                errors::InvalidArgument("bias1 dim mismatch: expected ",
                                        output_cols, ", got ",
                                        bias1.dim_size(0)));

    // The hidden tensor remains internal to this fused op. It is still
    // materialized as a temp buffer, but it is no longer exposed as a separate
    // graph output between two independently scheduled fused matmul nodes.
    Tensor hidden;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(input.dtype(), hidden_shape, &hidden));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    OP_REQUIRES_OK(ctx, RunMatMulWithBias(ctx, input, weight0, bias0,
                                          transpose_a0_, transpose_b0_,
                                          hidden_shape, &hidden));
    OP_REQUIRES_OK(ctx, ApplyReluInPlace(ctx, &hidden));

    const Tensor& second_lhs =
        hidden_input_idx1_ == 0 ? hidden : second_other;
    const Tensor& second_rhs =
        hidden_input_idx1_ == 0 ? second_other : hidden;
    OP_REQUIRES_OK(ctx, RunMatMulWithBias(ctx, second_lhs, second_rhs, bias1,
                                          transpose_a1_, transpose_b1_,
                                          output_shape, output));
  }

 private:
  Status RunMatMulWithBias(OpKernelContext* ctx, const Tensor& lhs,
                           const Tensor& rhs, const Tensor& bias,
                           bool trans_a, bool trans_b,
                           const TensorShape& out_shape, Tensor* out) {
    if (lhs.dims() == 2 && rhs.dims() == 2) {
      return Run2DMatMulWithBias(ctx, lhs, rhs, bias, trans_a, trans_b, out);
    }

    TF_RETURN_IF_ERROR(RunBatchMatMul(ctx, lhs, rhs, trans_a, trans_b, out));
    return ApplyBiasAddInPlace(ctx, bias, out_shape, out);
  }

  Status Run2DMatMulWithBias(OpKernelContext* ctx, const Tensor& lhs,
                             const Tensor& rhs, const Tensor& bias,
                             bool trans_a, bool trans_b, Tensor* out) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_lhs = CreateMTensor(lhs, format_);
    mTensor mt_rhs = CreateMTensor(rhs, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    mTensor mt_out = CreateMTensor(*out, format_);

    mMatMul op;
    op.SetTranspose(trans_a, trans_b);
    op.SetAlpha(1.0);
    op.SetBeta(0.0);

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      auto deleter = [tf_allocator](void* p) {
        if (p) {
          tf_allocator->DeallocateRaw(p);
        }
      };
      return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };
    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status = op.RunWithBiasAdd(handle, mt_out, mt_lhs, mt_rhs, mt_bias, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MatMul+BiasAdd epilogue failed. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status RunBatchMatMul(OpKernelContext* ctx, const Tensor& lhs,
                        const Tensor& rhs, bool trans_a, bool trans_b,
                        Tensor* out) {
    auto& handle = GetHandleByCtx(ctx);

    mBatchMatMul op;
    op.SetTranspose(trans_a, trans_b);
    op.SetAlpha(1.0);
    op.SetBeta(0.0);

    mTensor mt_lhs = CreateMTensor(lhs, format_);
    mTensor mt_rhs = CreateMTensor(rhs, format_);
    mTensor mt_out = CreateMTensor(*out, format_);

    std::vector<std::vector<int64_t>> shape_storage;
    shape_storage.reserve(6);

    auto FixToBatchFormat = [&](mTensor& mt, const Tensor& t) {
      if (t.dims() == 2) {
        const int64_t rows = t.dim_size(0);
        const int64_t cols = t.dim_size(1);
        std::vector<int64_t> dims = {1, rows, cols};
        std::vector<int64_t> strides = {rows * cols, cols, 1};
        shape_storage.push_back(std::move(dims));
        shape_storage.push_back(std::move(strides));
        mt.SetNdInfo(3, shape_storage[shape_storage.size() - 2].data(),
                     shape_storage[shape_storage.size() - 1].data());
      }
    };

    FixToBatchFormat(mt_lhs, lhs);
    FixToBatchFormat(mt_rhs, rhs);
    FixToBatchFormat(mt_out, *out);

    auto status = op.Run(handle, mt_out, mt_lhs, mt_rhs);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("BatchMatMul failed. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyBiasAddInPlace(OpKernelContext* ctx, const Tensor& bias,
                             const TensorShape& out_shape, Tensor* out) {
    auto& handle = GetHandleByCtx(ctx);
    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::ADD);

    mTensor mt_out = CreateMTensor(*out, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);

    const int dims_cnt = out_shape.dims();
    const int channel_dim = dims_cnt - 1;
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias.dim_size(0);
    b_strides[channel_dim] = 1;
    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    auto status = binary_op.Run(handle, mt_out, mt_out, mt_bias);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("BiasAdd failed. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyReluInPlace(OpKernelContext* ctx, Tensor* out) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*out, format_);

    mUnary unary_op;
    if (activation_type_ == "Relu") {
      unary_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    } else {
      unary_op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU);
      unary_op.SetAlpha(static_cast<double>(activation_alpha_));
    }

    auto status = unary_op.Run(handle, mt_out, mt_out);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("Activation failed. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  bool transpose_a0_ = false;
  bool transpose_b0_ = false;
  bool transpose_a1_ = false;
  bool transpose_b1_ = false;
  int hidden_input_idx1_ = 0;
  std::string activation_type_ = "Relu";
  float activation_alpha_ = 0.2f;
};

#define REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL(TYPE)                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MusaTwoLayerFusedMatMul")                                    \
          .Device("MUSA")                                                \
          .TypeConstraint<TYPE>("T"),                                    \
      MusaTwoLayerFusedMatMulOp<TYPE>);

REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL(float);
REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL(double);
REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL(Eigen::half);
REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL(bfloat16);

#undef REGISTER_MUSA_TWO_LAYER_FUSED_MATMUL

}  // namespace musa
}  // namespace tensorflow
