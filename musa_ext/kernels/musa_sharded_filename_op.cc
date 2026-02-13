#include <cstdio>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

class MusaShardedFilenameOp : public OpKernel {
 public:
  explicit MusaShardedFilenameOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& basename_t = ctx->input(0);
    const Tensor& shard_t = ctx->input(1);
    const Tensor& num_shards_t = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(basename_t.shape()),
                errors::InvalidArgument("basename must be a scalar"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(shard_t.shape()),
                errors::InvalidArgument("shard must be a scalar"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_shards_t.shape()),
                errors::InvalidArgument("num_shards must be a scalar"));

    const tstring& basename = basename_t.scalar<tstring>()();
    int32 shard = shard_t.scalar<int32>()();
    int32 num_shards = num_shards_t.scalar<int32>()();

    OP_REQUIRES(ctx, shard >= 0, errors::InvalidArgument("shard must be >= 0"));
    OP_REQUIRES(ctx, num_shards > 0,
                errors::InvalidArgument("num_shards must be > 0"));
    OP_REQUIRES(ctx, shard < num_shards,
                errors::InvalidArgument("shard must be < num_shards"));

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, basename_t.shape(), &output_t));

    char buffer[128];
    int n =
        snprintf(buffer, sizeof(buffer), "-%05d-of-%05d", shard, num_shards);

    OP_REQUIRES(ctx, n >= 0 && n < sizeof(buffer),
                errors::Internal("Formatting filename failed."));

    output_t->scalar<tstring>()() = basename + string(buffer);
  }
};

#define REGISTER_MUSA_KERNEL()                          \
  REGISTER_KERNEL_BUILDER(Name("ShardedFilename")       \
                              .Device("MUSA")           \
                              .HostMemory("basename")   \
                              .HostMemory("shard")      \
                              .HostMemory("num_shards") \
                              .HostMemory("filename"),  \
                          MusaShardedFilenameOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
