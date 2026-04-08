#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// Fused op for MusaConcatMatMul, which computes:
//   ConcatV2 + MatMul
//   ConcatV2 + MatMul + BiasAdd
template <typename T>
class MusaConcatMatMulOp : public MusaOpKernel {
 public:
  explicit MusaConcatMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_concat", &num_concat_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_input_idx", &concat_input_idx_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_args", &num_args_));

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));
    if (fused_ops.size() == 1 && fused_ops[0] == "BiasAdd" && num_args_ == 1) {
      fusion_type_ = FusionType::BIAS_ADD;
    } else {
      fusion_type_ = FusionType::NONE;
    }

    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return false;  // Default: TF32 disabled for higher precision
    }();
    tf32_enabled_ = tf32_enabled_global;
  }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    // 1. Get axis and concat inputs
    const Tensor& axis_tensor = ctx->input(num_concat_);
    int64 axis_val =
        axis_tensor.scalar<int32>()();  // Assuming Tidx=int32 for now

    std::vector<const Tensor*> concat_inputs;
    int64_t concat_dim_total = 0;
    int first_non_empty_idx = -1;

    for (int i = 0; i < num_concat_; ++i) {
      const Tensor& t = ctx->input(i);
      concat_inputs.push_back(&t);
      if (t.NumElements() > 0) {
        if (first_non_empty_idx == -1) first_non_empty_idx = i;
      }
    }

    const Tensor& ref =
        ctx->input(first_non_empty_idx == -1 ? 0 : first_non_empty_idx);
    int normalized_axis = axis_val < 0 ? axis_val + ref.dims() : axis_val;

    for (int i = 0; i < num_concat_; ++i) {
      concat_dim_total += ctx->input(i).dim_size(normalized_axis);
    }

    TensorShape concat_shape = ref.shape();
    concat_shape.set_dim(normalized_axis, concat_dim_total);

    // 2. Perform Concat (into temp)
    Tensor concat_out_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ref.dtype(), concat_shape, &concat_out_tensor));

    std::vector<::musa::dnn::Tensor> mudnn_ins;
    for (int i = 0; i < num_concat_; ++i) {
      if (ctx->input(i).NumElements() > 0) {
        mudnn_ins.push_back(CreateMTensor(ctx->input(i)));
      }
    }
    ::musa::dnn::Tensor mudnn_concat_out = CreateMTensor(concat_out_tensor);
    ::musa::dnn::Concat concat_op;
    concat_op.SetAxis(normalized_axis);
    auto status = concat_op.Run(handle, mudnn_concat_out, mudnn_ins.size(),
                                mudnn_ins.data());
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Concat failed in MusaConcatMatMul."));

    // 3. MatMul / MatMul+BiasAdd
    const Tensor& other_input = ctx->input(num_concat_ + 1);
    const Tensor& in0 =
        (concat_input_idx_ == 0) ? concat_out_tensor : other_input;
    const Tensor& in1 =
        (concat_input_idx_ == 1) ? concat_out_tensor : other_input;

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for MatMul in MusaConcatMatMul"));

    int64 m =
        trans_a_ ? in0.dim_size(in0.dims() - 1) : in0.dim_size(in0.dims() - 2);
    int64 n =
        trans_b_ ? in1.dim_size(in1.dims() - 2) : in1.dim_size(in1.dims() - 1);

    const Tensor* bias = nullptr;
    if (fusion_type_ == FusionType::BIAS_ADD) {
      OP_REQUIRES(ctx, num_args_ == 1,
                  errors::InvalidArgument(
                      "MusaConcatMatMul BiasAdd expects exactly 1 fused arg"));
      OP_REQUIRES(ctx, ctx->num_inputs() >= num_concat_ + 3,
                  errors::InvalidArgument(
                      "MusaConcatMatMul BiasAdd missing fused bias input"));
      bias = &ctx->input(num_concat_ + 2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias->shape()),
                  errors::InvalidArgument(
                      "Bias must be 1-D in MusaConcatMatMul, got shape ",
                      bias->shape().DebugString()));
      OP_REQUIRES(ctx, bias->dim_size(0) == n,
                  errors::InvalidArgument("Bias size mismatch in "
                                          "MusaConcatMatMul: expected ",
                                          n, ", got ", bias->dim_size(0)));
    }

    TensorShape mm_out_shape = bcast.output_batch_shape();
    mm_out_shape.AddDim(m);
    mm_out_shape.AddDim(n);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));

    if (fusion_type_ == FusionType::BIAS_ADD && in0.dims() == 2 &&
        in1.dims() == 2) {
      OP_REQUIRES_OK(ctx,
                     Run2DMatMulWithBias(ctx, in0, in1, *bias, output));
    } else {
      OP_REQUIRES_OK(ctx, RunMatMul(ctx, in0, in1, output));
      if (fusion_type_ == FusionType::BIAS_ADD) {
        OP_REQUIRES_OK(ctx, ApplyBiasAddInPlace(ctx, *bias, mm_out_shape, output));
      }
    }
  }

 private:
  Status RunMatMul(OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
                   Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);

    ::musa::dnn::Status status;
    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    } else {
      mBatchMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    }

    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA MatMul failed in MusaConcatMatMul. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status Run2DMatMulWithBias(OpKernelContext* ctx, const Tensor& a,
                             const Tensor& b, const Tensor& bias,
                             Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_a = CreateMTensor(a);
    mTensor mt_b = CreateMTensor(b);
    mTensor mt_bias = CreateMTensor(bias);
    mTensor mt_out = CreateMTensor(*output);

    mMatMul op;
    op.SetTranspose(trans_a_, trans_b_);
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

    auto status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA MatMul+BiasAdd epilogue failed in MusaConcatMatMul. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyBiasAddInPlace(OpKernelContext* ctx, const Tensor& bias,
                             const TensorShape& out_shape, Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output);
    mTensor mt_bias = CreateMTensor(bias);

    const int dims_cnt = out_shape.dims();
    const int channel_dim = dims_cnt - 1;
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias.dim_size(0);
    b_strides[channel_dim] = 1;
    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    mBinary bias_add_op;
    bias_add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    auto status = bias_add_op.Run(handle, mt_out, mt_out, mt_bias);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA BiasAdd failed in MusaConcatMatMul. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  enum class FusionType { NONE, BIAS_ADD };

  bool trans_a_ = false;
  bool trans_b_ = false;
  int num_concat_ = 0;
  int concat_input_idx_ = 0;
  int num_args_ = 0;
  bool tf32_enabled_ = false;
  FusionType fusion_type_ = FusionType::NONE;
};

#define REGISTER_MUSA_CONCAT_MATMUL(TYPE)                \
  REGISTER_KERNEL_BUILDER(Name("MusaConcatMatMul")       \
                              .Device("MUSA")            \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("axis"),       \
                          MusaConcatMatMulOp<TYPE>);

REGISTER_MUSA_CONCAT_MATMUL(float);
REGISTER_MUSA_CONCAT_MATMUL(Eigen::half);
REGISTER_MUSA_CONCAT_MATMUL(double);
REGISTER_MUSA_CONCAT_MATMUL(bfloat16);

}  // namespace musa

REGISTER_OP("MusaConcatMatMul")
    .Input("inputs: num_concat * T")
    .Input("axis: int32")
    .Input("other: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {float, half, bfloat16, double}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("num_concat: int >= 1")
    .Attr("concat_input_idx: int")
    .Attr("fused_ops: list(string) = []")
    .Attr("num_args: int >= 0 = 0")
    .SetShapeFn(shape_inference::MatMulShape);

}  // namespace tensorflow
