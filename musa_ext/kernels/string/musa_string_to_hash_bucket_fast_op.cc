#include <musa_runtime.h>

#include "kernels/utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace musa {

class MusaStringToHashBucketFastOp : public OpKernel {
 public:
  explicit MusaStringToHashBucketFastOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    if (input_tensor->NumElements() == 0) return;

    int64 N = input_tensor->NumElements();

    std::vector<int64> host_output(N);

    for (int64 i = 0; i < N; ++i) {
      const tstring& s = input_flat(i);

      uint64 hash = tensorflow::Hash64(s.data(), s.size());

      host_output[i] = static_cast<int64>(hash % num_buckets_);
    }

    int64* device_ptr = output_tensor->flat<int64>().data();

    // Use async memcpy with stream for better concurrency
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    mStatus status = MusaMemcpyAsyncH2D(device_ptr, host_output.data(),
                                        N * sizeof(int64), stream);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA StringToHashBucketFast: memcpy failed"));

    // Synchronize only the current stream
    if (stream) {
      musaStreamSynchronize(stream);
    }
  }

 private:
  int64 num_buckets_;
};

#define REGISTER_MUSA_KERNEL()                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("StringToHashBucketFast").Device("MUSA").HostMemory("input"), \
      MusaStringToHashBucketFastOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
