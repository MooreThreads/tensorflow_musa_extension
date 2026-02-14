#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace musa {

class MusaConcatOffsetOp : public OpKernel {
 public:
  explicit MusaConcatOffsetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& concat_dim_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(concat_dim_tensor.shape()),
                errors::InvalidArgument(
                    "Concat dim tensor should be a scalar input, but is ",
                    concat_dim_tensor.shape().DebugString()));
    int32 concat_dim = concat_dim_tensor.scalar<int32>()();

    const int N = ctx->num_inputs() - 1;
    const Tensor& inp0 = ctx->input(1);
    auto inp0_vec = inp0.vec<int32>();
    const int rank = inp0.NumElements();

    if (concat_dim < 0) concat_dim += rank;
    OP_REQUIRES(ctx, concat_dim >= 0 && concat_dim < rank,
                errors::InvalidArgument(
                    "Concat dim is out of range: ", concat_dim, " vs. ", rank));

    std::vector<int32> offset(rank, 0);

    for (int i = 0; i < N; ++i) {
      const Tensor& inp = ctx->input(1 + i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(inp.shape()),
                  errors::InvalidArgument("Input shape should be a vector."));
      OP_REQUIRES(
          ctx, inp.NumElements() == rank,
          errors::InvalidArgument("All input shapes must have the same rank."));

      auto inp_vec = inp.vec<int32>();

      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, TensorShape({rank}), &out));
      auto out_vec = out->vec<int32>();

      for (int r = 0; r < rank; ++r) {
        out_vec(r) = offset[r];
        if (r == concat_dim) {
          offset[r] += inp_vec(r);
        } else {
          OP_REQUIRES(ctx, inp_vec(r) == inp0_vec(r),
                      errors::InvalidArgument("Input shapes must match in all "
                                              "dimensions except concat_dim."));
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ConcatOffset")
                            .Device("MUSA")
                            .HostMemory("concat_dim")
                            .HostMemory("shape")
                            .HostMemory("offset"),
                        MusaConcatOffsetOp);

}  // namespace musa
}  // namespace tensorflow
