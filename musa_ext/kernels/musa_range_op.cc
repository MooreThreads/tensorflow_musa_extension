#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>

#include "kernels/utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

// ============================================================================
// MUSA Range custom kernel launcher declarations from musa_range_kernel.mu
// ============================================================================

extern "C" {
void LaunchRangeKernelFloat(float* output, float start, float delta, int size,
                            musaStream_t stream);
void LaunchRangeKernelDouble(double* output, double start, double delta,
                             int size, musaStream_t stream);
void LaunchRangeKernelInt32(int* output, int start, int delta, int size,
                            musaStream_t stream);
void LaunchRangeKernelInt64(long long* output, long long start, long long delta,
                            int size, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// Optimized Range implementation using custom MUSA kernel
//
// PERFORMANCE COMPARISON:
//
// Original (host computation + H2D copy):
//   - Host loop to generate values
//   - Host memory allocation (std::vector)
//   - H2D memory transfer (PCIe bottleneck)
//   - Host stream synchronization
//
// Optimized (device kernel):
//   - Single kernel launch
//   - No host memory allocation
//   - No H2D transfer
//   - Fully parallel on device
//
// EXPECTED SPEEDUP: 5-10x (eliminates PCIe transfer)

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

    // OPTIMIZED: Launch custom kernel to generate values on device
    auto device_ptr = output->flat<T>().data();
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    LaunchRangeKernel(ctx, device_ptr, start, delta, static_cast<int>(size),
                      stream);
  }

 private:
  // Helper to call the correct launcher function for each type
  void LaunchRangeKernel(OpKernelContext* ctx, T* output, T start, T delta,
                         int size, musaStream_t stream);
};

// ============================================================================
// Template specializations using macro to reduce boilerplate
// ============================================================================

#define DEFINE_RANGE_SPECIALIZATION(T, launcher)                          \
  template <>                                                             \
  void MusaRangeOp<T>::LaunchRangeKernel(OpKernelContext* ctx, T* output, \
                                         T start, T delta, int size,      \
                                         musaStream_t stream) {           \
    launcher(output, start, delta, size, stream);                         \
  }

DEFINE_RANGE_SPECIALIZATION(float, LaunchRangeKernelFloat)
DEFINE_RANGE_SPECIALIZATION(double, LaunchRangeKernelDouble)
DEFINE_RANGE_SPECIALIZATION(int32, LaunchRangeKernelInt32)
DEFINE_RANGE_SPECIALIZATION(int64, LaunchRangeKernelInt64)

#undef DEFINE_RANGE_SPECIALIZATION

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
