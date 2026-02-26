#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

// ============================================================================
// MUSA Softplus custom kernel launcher declarations from
// musa_softplus_kernel.mu
// ============================================================================

extern "C" {
void LaunchSoftplusKernelFloat(const float* input, float* output, int size,
                               musaStream_t stream);
void LaunchSoftplusKernelDouble(const double* input, double* output, int size,
                                musaStream_t stream);
void LaunchSoftplusKernelHalf(const void* input, void* output, int size,
                              musaStream_t stream);
void LaunchSoftplusKernelBFloat16(const void* input, void* output, int size,
                                  musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ============================================================================
// Common implementation for Softplus Compute
// ============================================================================

template <typename T>
void SoftplusCompute(OpKernelContext* ctx,
                     void (*launcher)(const T*, T*, int, musaStream_t)) {
  OP_REQUIRES(ctx, ctx->num_inputs() == 1,
              errors::InvalidArgument("Softplus expects 1 input, got ",
                                      ctx->num_inputs()));

  const Tensor& input = ctx->input(0);
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

  const int64 num_elements = input.NumElements();
  if (num_elements == 0) return;

  musaStream_t stream =
      reinterpret_cast<musaStream_t>(GetHandleByCtx(ctx).GetStream());

  const void* input_ptr = input.tensor_data().data();
  void* output_ptr = const_cast<char*>(output->tensor_data().data());

  launcher(reinterpret_cast<const T*>(input_ptr),
           reinterpret_cast<T*>(output_ptr), static_cast<int>(num_elements),
           stream);
}

// ============================================================================
// Softplus operator class
// ============================================================================

template <typename T>
class MusaSoftplusOp : public MusaOpKernel {
 public:
  explicit MusaSoftplusOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SoftplusCompute<T>(ctx, GetLauncher());
  }

 private:
  static void (*GetLauncher())(const T*, T*, int, musaStream_t);
};

// ============================================================================
// Launcher function getters - specialized for each type
// ============================================================================

#define DEFINE_SOFTPLUS_LAUNCHER_GETTER(T, launcher, input_cast, output_cast) \
  template <>                                                                 \
  void (*MusaSoftplusOp<T>::GetLauncher())(const T*, T*, int, musaStream_t) { \
    return [](const T* input, T* output, int size, musaStream_t stream) {     \
      launcher(input_cast(input), output_cast(output), size, stream);         \
    };                                                                        \
  }

// Float / double
DEFINE_SOFTPLUS_LAUNCHER_GETTER(float, LaunchSoftplusKernelFloat,
                                reinterpret_cast<const float*>,
                                reinterpret_cast<float*>)

DEFINE_SOFTPLUS_LAUNCHER_GETTER(double, LaunchSoftplusKernelDouble,
                                reinterpret_cast<const double*>,
                                reinterpret_cast<double*>)

// Half / BFloat16 use void* bridge
DEFINE_SOFTPLUS_LAUNCHER_GETTER(Eigen::half, LaunchSoftplusKernelHalf,
                                reinterpret_cast<const void*>,
                                reinterpret_cast<void*>)

DEFINE_SOFTPLUS_LAUNCHER_GETTER(bfloat16, LaunchSoftplusKernelBFloat16,
                                reinterpret_cast<const void*>,
                                reinterpret_cast<void*>)

#undef DEFINE_SOFTPLUS_LAUNCHER_GETTER

// ============================================================================
// Kernel registration (TF2 only)
// Name must match TensorFlow official op: "Softplus"
// ============================================================================

#define REGISTER_MUSA_SOFTPLUS(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Softplus").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaSoftplusOp<TYPE>);

REGISTER_MUSA_SOFTPLUS(float);
REGISTER_MUSA_SOFTPLUS(double);
// REGISTER_MUSA_SOFTPLUS(Eigen::half);
// REGISTER_MUSA_SOFTPLUS(bfloat16);

#undef REGISTER_MUSA_SOFTPLUS

}  // namespace musa
}  // namespace tensorflow