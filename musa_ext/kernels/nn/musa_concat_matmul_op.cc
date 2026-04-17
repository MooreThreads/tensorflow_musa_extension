#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// Fused op for MusaConcatMatMul, which computes ConcatV2 + MatMul
template <typename T>
class MusaConcatMatMulOp : public MusaOpKernel {
 public:
  explicit MusaConcatMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_concat", &num_concat_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_input_idx", &concat_input_idx_));

    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return true;  // Default: TF32 enabled for performance
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

    // 3. MatMul
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

    TensorShape mm_out_shape = bcast.output_batch_shape();
    mm_out_shape.AddDim(m);
    mm_out_shape.AddDim(n);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm_out_shape, &output));

    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    } else {
      mBatchMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    }

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA MatMul failed in MusaConcatMatMul."));
  }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
  int num_concat_ = 0;
  int concat_input_idx_ = 0;
  bool tf32_enabled_ = true;
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
    .Output("output: T")
    .Attr("T: {float, half, bfloat16, double}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("num_concat: int >= 1")
    .Attr("concat_input_idx: int")
    .SetShapeFn(shape_inference::MatMulShape);

}  // namespace tensorflow
