#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {
namespace musa {

namespace {

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
//   ConcatV2 + MatMul + BiasAdd + Relu + MatMul + BiasAdd
//   ConcatV2 + MatMul + BiasAdd + LeakyRelu + MatMul + BiasAdd
//
// This op keeps the Concat stage inside the fused op boundary, then executes
// two linear stages with a single activation in between.
REGISTER_OP("MusaTwoLayerConcatMatMul")
    .Input("inputs: num_concat * T")
    .Input("axis: int32")
    .Input("other0: T")
    .Input("bias0: T")
    .Input("other1: T")
    .Input("bias1: T")
    .Output("output: T")
    .Attr("T: {float, half, double, bfloat16}")
    .Attr("num_concat: int >= 1")
    .Attr("concat_input_idx: int")
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
class MusaTwoLayerConcatMatMulOp : public MusaOpKernel {
 public:
  explicit MusaTwoLayerConcatMatMulOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_concat", &num_concat_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_input_idx", &concat_input_idx_));
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
    OP_REQUIRES(ctx,
                activation_type_ == "Relu" || activation_type_ == "LeakyRelu",
                errors::InvalidArgument(
                    "activation_type must be Relu or LeakyRelu, got ",
                    activation_type_));

    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return false;
    }();
    tf32_enabled_ = tf32_enabled_global;
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    const Tensor& axis_tensor = ctx->input(num_concat_);
    int64 axis_val = axis_tensor.scalar<int32>()();

    int first_non_empty_idx = -1;
    for (int i = 0; i < num_concat_; ++i) {
      if (ctx->input(i).NumElements() > 0) {
        first_non_empty_idx = i;
        break;
      }
    }
    const Tensor& ref =
        ctx->input(first_non_empty_idx == -1 ? 0 : first_non_empty_idx);
    int normalized_axis = axis_val < 0 ? axis_val + ref.dims() : axis_val;

    int64_t concat_dim_total = 0;
    std::vector<::musa::dnn::Tensor> mudnn_ins;
    for (int i = 0; i < num_concat_; ++i) {
      const Tensor& t = ctx->input(i);
      concat_dim_total += t.dim_size(normalized_axis);
      if (t.NumElements() > 0) {
        mudnn_ins.push_back(CreateMTensor(t));
      }
    }

    TensorShape concat_shape = ref.shape();
    concat_shape.set_dim(normalized_axis, concat_dim_total);
    Tensor concat_out;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(ref.dtype(), concat_shape, &concat_out));

    ::musa::dnn::Tensor mt_concat_out = CreateMTensor(concat_out);
    ::musa::dnn::Concat concat_op;
    concat_op.SetAxis(normalized_axis);
    auto status = concat_op.Run(handle, mt_concat_out, mudnn_ins.size(),
                                mudnn_ins.data());
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA Concat failed in MusaTwoLayerConcatMatMul."));

    const Tensor& other0 = ctx->input(num_concat_ + 1);
    const Tensor& bias0 = ctx->input(num_concat_ + 2);
    const Tensor& other1 = ctx->input(num_concat_ + 3);
    const Tensor& bias1 = ctx->input(num_concat_ + 4);

    const Tensor& first_lhs = concat_input_idx_ == 0 ? concat_out : other0;
    const Tensor& first_rhs = concat_input_idx_ == 0 ? other0 : concat_out;

    TensorShape hidden_shape;
    int64 hidden_cols = 0;
    OP_REQUIRES_OK(ctx, InferMatMulOutputShape(first_lhs.shape(),
                                               first_rhs.shape(), transpose_a0_,
                                               transpose_b0_, &hidden_shape,
                                               &hidden_cols));
    OP_REQUIRES(ctx, bias0.dims() == 1,
                errors::InvalidArgument("bias0 must be 1D"));
    OP_REQUIRES(ctx, bias0.dim_size(0) == hidden_cols,
                errors::InvalidArgument("bias0 dim mismatch: expected ",
                                        hidden_cols, ", got ",
                                        bias0.dim_size(0)));

    TensorShape output_shape;
    int64 output_cols = 0;
    const TensorShape& second_lhs_shape =
        hidden_input_idx1_ == 0 ? hidden_shape : other1.shape();
    const TensorShape& second_rhs_shape =
        hidden_input_idx1_ == 0 ? other1.shape() : hidden_shape;
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

    Tensor hidden;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(ref.dtype(), hidden_shape, &hidden));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    OP_REQUIRES_OK(ctx, RunMatMulWithBias(ctx, first_lhs, first_rhs, bias0,
                                          transpose_a0_, transpose_b0_,
                                          hidden_shape, &hidden));
    OP_REQUIRES_OK(ctx, ApplyActivationInPlace(ctx, &hidden));

    const Tensor& second_lhs = hidden_input_idx1_ == 0 ? hidden : other1;
    const Tensor& second_rhs = hidden_input_idx1_ == 0 ? other1 : hidden;
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
      return errors::Internal(
          "MatMul+BiasAdd epilogue failed in MusaTwoLayerConcatMatMul. Status: ",
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
      return errors::Internal(
          "BatchMatMul failed in MusaTwoLayerConcatMatMul. Status: ",
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
      return errors::Internal(
          "BiasAdd failed in MusaTwoLayerConcatMatMul. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyActivationInPlace(OpKernelContext* ctx, Tensor* out) {
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
      return errors::Internal(
          "Activation failed in MusaTwoLayerConcatMatMul. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  int num_concat_ = 0;
  int concat_input_idx_ = 0;
  bool transpose_a0_ = false;
  bool transpose_b0_ = false;
  bool transpose_a1_ = false;
  bool transpose_b1_ = false;
  int hidden_input_idx1_ = 0;
  std::string activation_type_ = "Relu";
  float activation_alpha_ = 0.2f;
  bool tf32_enabled_ = false;
};

#define REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL(TYPE)                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MusaTwoLayerConcatMatMul")                                    \
          .Device("MUSA")                                                 \
          .TypeConstraint<TYPE>("T")                                      \
          .HostMemory("axis"),                                            \
      MusaTwoLayerConcatMatMulOp<TYPE>);

REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL(float);
REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL(double);
REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL(Eigen::half);
REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL(bfloat16);

#undef REGISTER_MUSA_TWO_LAYER_CONCAT_MATMUL

}  // namespace musa
}  // namespace tensorflow
