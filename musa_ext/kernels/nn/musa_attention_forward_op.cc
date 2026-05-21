#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {
namespace {

extern "C" void LaunchMusaCausalAttentionForwardFloat(
    const float* query, const float* key, const float* value, const float* scale,
    float* softmax, float* output, bool store_softmax, int64_t groups,
    int query_dim, int key_dim, int head_dim, musaStream_t stream);
extern "C" void LaunchMusaCausalAttentionForwardBFloat16(
    const bfloat16* query, const bfloat16* key, const bfloat16* value,
    const bfloat16* scale, bfloat16* softmax, bfloat16* output,
    bool store_softmax, int64_t groups, int query_dim, int key_dim,
    int head_dim, musaStream_t stream);

class MusaCausalAttentionForwardOp : public MusaOpKernel {
 public:
  explicit MusaCausalAttentionForwardOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("store_softmax", &store_softmax_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& query = ctx->input(0);
    const Tensor& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    const Tensor& scale = ctx->input(3);
    OP_REQUIRES(ctx, (query.dtype() == DT_FLOAT ||
                      query.dtype() == DT_BFLOAT16) &&
                         key.dtype() == query.dtype() &&
                         value.dtype() == query.dtype() &&
                         scale.dtype() == query.dtype(),
                errors::InvalidArgument(
                    "MusaCausalAttentionForward expects float/bfloat16 inputs"));
    OP_REQUIRES(ctx, scale.NumElements() == 1,
                errors::InvalidArgument(
                    "MusaCausalAttentionForward expects scalar scale"));
    OP_REQUIRES(ctx, query.dims() == 4 && key.dims() == 4 && value.dims() == 4,
                errors::InvalidArgument(
                    "MusaCausalAttentionForward expects rank-4 inputs"));

    const int64_t batch = query.dim_size(0);
    const int64_t heads = query.dim_size(1);
    const int64_t query_dim = query.dim_size(2);
    const int64_t key_dim = key.dim_size(2);
    const int64_t head_dim = query.dim_size(3);
    OP_REQUIRES(ctx, key.dim_size(0) == batch && key.dim_size(1) == heads &&
                         value.dim_size(0) == batch &&
                         value.dim_size(1) == heads &&
                         value.dim_size(2) == key_dim &&
                         key.dim_size(3) == head_dim &&
                         value.dim_size(3) == head_dim,
                errors::InvalidArgument(
                    "MusaCausalAttentionForward incompatible shapes"));
    OP_REQUIRES(ctx, query_dim > 0 && key_dim > 0 && head_dim == 8 &&
                         query_dim <= key_dim && key_dim <= 64,
                errors::InvalidArgument(
                    "MusaCausalAttentionForward invalid shape"));

    TensorShape softmax_shape;
    if (store_softmax_) {
      softmax_shape.AddDim(batch);
      softmax_shape.AddDim(heads);
      softmax_shape.AddDim(query_dim);
      softmax_shape.AddDim(key_dim);
    }
    Tensor* softmax = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, softmax_shape, &softmax));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, query.shape(), &output));
    if (query.NumElements() == 0 || key.NumElements() == 0) return;

    if (query.dtype() == DT_FLOAT) {
      LaunchMusaCausalAttentionForwardFloat(
          query.flat<float>().data(), key.flat<float>().data(),
          value.flat<float>().data(), scale.flat<float>().data(),
          softmax->flat<float>().data(), output->flat<float>().data(),
          store_softmax_, batch * heads, static_cast<int>(query_dim),
          static_cast<int>(key_dim), static_cast<int>(head_dim),
          GetMusaStreamByCtx(ctx));
    } else {
      LaunchMusaCausalAttentionForwardBFloat16(
          query.flat<bfloat16>().data(), key.flat<bfloat16>().data(),
          value.flat<bfloat16>().data(), scale.flat<bfloat16>().data(),
          softmax->flat<bfloat16>().data(), output->flat<bfloat16>().data(),
          store_softmax_, batch * heads, static_cast<int>(query_dim),
          static_cast<int>(key_dim), static_cast<int>(head_dim),
          GetMusaStreamByCtx(ctx));
    }
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal(
                    "MusaCausalAttentionForward kernel failed: ",
                    musaGetErrorString(status)));
  }

 private:
  bool store_softmax_ = true;
};

#define REGISTER_MUSA_CAUSAL_ATTENTION_FORWARD(TYPE)                 \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MusaCausalAttentionForward").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaCausalAttentionForwardOp);

REGISTER_MUSA_CAUSAL_ATTENTION_FORWARD(float);
REGISTER_MUSA_CAUSAL_ATTENTION_FORWARD(bfloat16);

#undef REGISTER_MUSA_CAUSAL_ATTENTION_FORWARD

}  // namespace
}  // namespace musa

REGISTER_OP("MusaCausalAttentionForward")
    .Input("query: T")
    .Input("key: T")
    .Input("value: T")
    .Input("scale: T")
    .Output("softmax: T")
    .Output("output: T")
    .Attr("T: {float, bfloat16}")
    .Attr("store_softmax: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle query;
      ::tensorflow::shape_inference::ShapeHandle key;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &query));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &key));
      bool store_softmax = true;
      TF_RETURN_IF_ERROR(c->GetAttr("store_softmax", &store_softmax));
      if (store_softmax) {
        c->set_output(0,
                      c->MakeShape({c->Dim(query, 0), c->Dim(query, 1),
                                    c->Dim(query, 2), c->Dim(key, 2)}));
      } else {
        c->set_output(0, c->Scalar());
      }
      c->set_output(1, query);
      return ::tensorflow::OkStatus();
    });

}  // namespace tensorflow
