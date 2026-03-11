#include <mudnn.h>

#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

template <typename Type>
class MusaTokenMixerOp : public MusaOpKernel {
 public:
  explicit MusaTokenMixerOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_T", &num_T_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_H", &num_H_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("d_k", &d_k_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    
    OP_REQUIRES(ctx, input.dims() == 3,
                errors::InvalidArgument("Input must be rank 3, got ",
                                        input.dims()));
    const int64 B = input.dim_size(0);
    const int64 T = input.dim_size(1);
    const int64 D = input.dim_size(2);
    OP_REQUIRES(ctx, T == num_T_,
                errors::InvalidArgument(
                    "dim 1 must be num_T=", num_T_, ", got ", T));
    OP_REQUIRES(ctx, D == num_H_ * d_k_,
                errors::InvalidArgument(
                    "dim 2 must be num_H*d_k=", num_H_ * d_k_, ", got ", D));

    // Output: (B, H, T*d_k)
    TensorShape output_shape({B, num_H_, num_T_ * d_k_});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // ---- Zero-copy 4D views for the Permute operation ----
    // Input view: (B, T, H, d_k) — first Reshape, no data movement
    Tensor input_4d(input.dtype());
    OP_REQUIRES(ctx,
                input_4d.CopyFrom(input, TensorShape({B, num_T_, num_H_, d_k_})),
                errors::Internal("Failed to create 4D view of input"));

    // Output view: (B, H, T, d_k) — before the final Reshape, no data movement
    Tensor output_4d(output->dtype());
    OP_REQUIRES(ctx,
                output_4d.CopyFrom(*output, TensorShape({B, num_H_, num_T_, d_k_})),
                errors::Internal("Failed to create 4D view of output"));

    // ---- Transpose [0,2,1,3] via muDNN Permute ----
    mTensor in_mt = CreateMTensor(input_4d);
    mTensor out_mt = CreateMTensor(output_4d);

    auto& handle = GetHandleByCtx(ctx);
    ::musa::dnn::Permute permute_op;
    std::vector<int64_t> perm = {0, 2, 1, 3};

    auto config_status = permute_op.ConfigDimStride(
        out_mt, in_mt, static_cast<int>(perm.size()), perm.data());
    OP_REQUIRES(ctx, config_status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN Permute ConfigDimStride failed"));

    auto run_status = permute_op.Run(handle, out_mt, in_mt);
    OP_REQUIRES(ctx, run_status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN Permute Run failed"));
  }

 private:
  int64 num_T_;
  int64 num_H_;
  int64 d_k_;
};

#define REGISTER_MUSA_TOKENMIXER(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MusaTokenMixer").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaTokenMixerOp<TYPE>);

REGISTER_MUSA_TOKENMIXER(float);
REGISTER_MUSA_TOKENMIXER(Eigen::half);
REGISTER_MUSA_TOKENMIXER(bfloat16);

#undef REGISTER_MUSA_TOKENMIXER

}  // namespace musa

REGISTER_OP("MusaTokenMixer")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("num_T: int")
    .Attr("num_H: int")
    .Attr("d_k: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int64 num_H, d_k, num_T;
      TF_RETURN_IF_ERROR(c->GetAttr("num_H", &num_H));
      TF_RETURN_IF_ERROR(c->GetAttr("d_k", &d_k));
      TF_RETURN_IF_ERROR(c->GetAttr("num_T", &num_T));

      auto batch_dim = c->Dim(c->input(0), 0);
      c->set_output(0, c->MakeShape({batch_dim,
                                      c->MakeDim(num_H),
                                      c->MakeDim(num_T * d_k)}));
      return Status::OK();
    });

}  // namespace tensorflow