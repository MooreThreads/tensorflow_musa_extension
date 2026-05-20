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

extern "C" void LaunchMusaCausalAttentionGradFloat(
    const float* softmax, const float* dout, const float* query,
    const float* key, const float* value, const float* scale, float* dquery,
    float* dkey, float* dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim, musaStream_t stream);
extern "C" void LaunchMusaCausalAttentionGradBFloat16(
    const bfloat16* softmax, const bfloat16* dout, const bfloat16* query,
    const bfloat16* key, const bfloat16* value, const bfloat16* scale,
    bfloat16* dquery, bfloat16* dkey, bfloat16* dvalue, int64_t groups,
    int query_dim, int key_dim, int value_dim, musaStream_t stream);

class MusaCausalAttentionGradOp : public MusaOpKernel {
 public:
  explicit MusaCausalAttentionGradOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& softmax = ctx->input(0);
    const Tensor& dout = ctx->input(1);
    const Tensor& query = ctx->input(2);
    const Tensor& key = ctx->input(3);
    const Tensor& value = ctx->input(4);
    const Tensor& scale = ctx->input(5);
    OP_REQUIRES(ctx, (query.dtype() == DT_FLOAT ||
                      query.dtype() == DT_BFLOAT16) &&
                         softmax.dtype() == query.dtype() &&
                         dout.dtype() == query.dtype() &&
                         key.dtype() == query.dtype() &&
                         value.dtype() == query.dtype() &&
                         scale.dtype() == query.dtype(),
                errors::InvalidArgument(
                    "MusaCausalAttentionGrad expects float/bfloat16"));
    OP_REQUIRES(ctx, scale.NumElements() == 1,
                errors::InvalidArgument("MusaCausalAttentionGrad expects scalar scale"));
    OP_REQUIRES(ctx, (softmax.dims() == 4 || softmax.dims() == 0) &&
                         dout.dims() == 4 && query.dims() == 4 &&
                         key.dims() == 4 && value.dims() == 4,
                errors::InvalidArgument("MusaCausalAttentionGrad expects rank-4 tensors"));

    const int64_t batch = query.dim_size(0);
    const int64_t heads = query.dim_size(1);
    const int64_t query_dim = query.dim_size(2);
    const int64_t key_dim = key.dim_size(2);
    const int64_t value_dim = value.dim_size(3);
    if (softmax.dims() == 4) {
      OP_REQUIRES(ctx, softmax.dim_size(0) == batch &&
                           softmax.dim_size(1) == heads &&
                           softmax.dim_size(2) == query_dim &&
                           softmax.dim_size(3) == key_dim,
                  errors::InvalidArgument("MusaCausalAttentionGrad incompatible softmax shape"));
    }
    OP_REQUIRES(ctx, dout.dim_size(0) == batch && dout.dim_size(1) == heads &&
                         dout.dim_size(2) == query_dim &&
                         dout.dim_size(3) == value_dim &&
                         query.dim_size(3) == value_dim &&
                         key.dim_size(0) == batch && key.dim_size(1) == heads &&
                         key.dim_size(3) == value_dim &&
                         value.dim_size(0) == batch && value.dim_size(1) == heads &&
                         value.dim_size(2) == key_dim,
                errors::InvalidArgument("MusaCausalAttentionGrad incompatible shapes"));
    OP_REQUIRES(ctx, query_dim > 0 && key_dim > 0 && value_dim > 0 &&
                         query_dim <= key_dim && query_dim <= 64 &&
                         key_dim <= 64 && value_dim <= 64,
                errors::InvalidArgument("MusaCausalAttentionGrad invalid shape"));

    Tensor* dquery = nullptr;
    Tensor* dkey = nullptr;
    Tensor* dvalue = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &dquery));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, key.shape(), &dkey));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, value.shape(), &dvalue));
    if (dquery->NumElements() == 0) return;

    if (query.dtype() == DT_FLOAT) {
      LaunchMusaCausalAttentionGradFloat(
          softmax.flat<float>().data(), dout.flat<float>().data(),
          query.flat<float>().data(), key.flat<float>().data(),
          value.flat<float>().data(), scale.flat<float>().data(),
          dquery->flat<float>().data(), dkey->flat<float>().data(),
          dvalue->flat<float>().data(), batch * heads,
          static_cast<int>(query_dim), static_cast<int>(key_dim),
          static_cast<int>(value_dim), GetMusaStreamByCtx(ctx));
    } else {
      LaunchMusaCausalAttentionGradBFloat16(
          softmax.flat<bfloat16>().data(), dout.flat<bfloat16>().data(),
          query.flat<bfloat16>().data(), key.flat<bfloat16>().data(),
          value.flat<bfloat16>().data(), scale.flat<bfloat16>().data(),
          dquery->flat<bfloat16>().data(), dkey->flat<bfloat16>().data(),
          dvalue->flat<bfloat16>().data(), batch * heads,
          static_cast<int>(query_dim), static_cast<int>(key_dim),
          static_cast<int>(value_dim), GetMusaStreamByCtx(ctx));
    }
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaCausalAttentionGrad kernel failed: ",
                                 musaGetErrorString(status)));
  }
};

#define REGISTER_MUSA_CAUSAL_ATTENTION_GRAD(TYPE)                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MusaCausalAttentionGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaCausalAttentionGradOp);

REGISTER_MUSA_CAUSAL_ATTENTION_GRAD(float);
REGISTER_MUSA_CAUSAL_ATTENTION_GRAD(bfloat16);

#undef REGISTER_MUSA_CAUSAL_ATTENTION_GRAD

}  // namespace
}  // namespace musa

REGISTER_OP("MusaCausalAttentionGrad")
    .Input("softmax: T")
    .Input("dout: T")
    .Input("query: T")
    .Input("key: T")
    .Input("value: T")
    .Input("scale: T")
    .Output("dquery: T")
    .Output("dkey: T")
    .Output("dvalue: T")
    .Attr("T: {float, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      return ::tensorflow::OkStatus();
    });

}  // namespace tensorflow
