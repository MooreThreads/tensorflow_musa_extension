#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaRangeOp : public OpKernel {
 public:
  explicit MusaRangeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& start_in = ctx->input(0);
    const Tensor& limit_in = ctx->input(1);
    const Tensor& delta_in = ctx->input(2);

    T start = start_in.scalar<T>()();
    T limit = limit_in.scalar<T>()();
    T delta = delta_in.scalar<T>()();

    OP_REQUIRES(ctx, delta != static_cast<T>(0),
                errors::InvalidArgument("Requires delta != 0"));

    if (delta > 0) {
      OP_REQUIRES(
          ctx, start <= limit,
          errors::InvalidArgument("Requires start <= limit when delta > 0"));
    } else {
      OP_REQUIRES(
          ctx, start >= limit,
          errors::InvalidArgument("Requires start >= limit when delta < 0"));
    }

    int64 size = 0;
    if ((delta > 0 && start < limit) || (delta < 0 && start > limit)) {
      size = static_cast<int64>(
          std::ceil(std::abs((double)(limit - start) / (double)delta)));
    }

    TensorShape output_shape;
    output_shape.AddDim(size);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (size == 0) return;

    std::vector<T> host_data(size);
    T val = start;
    for (int64 i = 0; i < size; ++i) {
      host_data[i] = val;
      val += delta;
    }

    auto device_ptr = output->flat<T>().data();
    auto status = musaMemcpy(device_ptr, host_data.data(), size * sizeof(T),
                             musaMemcpyHostToDevice);

    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaRangeOp: musaMemcpy failed. Error code: ",
                                 static_cast<int>(status)));
  }
};

#define REGISTER_RANGE(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("Range")                   \
                              .Device("MUSA")             \
                              .HostMemory("start")        \
                              .HostMemory("limit")        \
                              .HostMemory("delta")        \
                              .TypeConstraint<T>("Tidx"), \
                          MusaRangeOp<T>);

REGISTER_RANGE(float);
REGISTER_RANGE(double);
REGISTER_RANGE(int32);
REGISTER_RANGE(int64);

#undef REGISTER_RANGE

}  // namespace musa
}  // namespace tensorflow
