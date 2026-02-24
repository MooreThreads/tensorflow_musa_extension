#include <cstdint>
#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
// Launches the device kernel that computes y[i] = is_nan(x[i]).
void LaunchIsNan(const T* input, bool* output, int n, musaStream_t stream);
void LaunchIsNanHalf(const uint16_t* input, bool* output, int n,
                     musaStream_t stream);

template <typename T>
struct IsNanLaunchHelper {
  static void Run(const T* input, bool* output, int n, musaStream_t stream) {
    LaunchIsNan<T>(input, output, n, stream);
  }
};

template <>
struct IsNanLaunchHelper<Eigen::half> {
  static void Run(const Eigen::half* input, bool* output, int n,
                  musaStream_t stream) {
    LaunchIsNanHalf(reinterpret_cast<const uint16_t*>(input), output, n,
                    stream);
  }
};

template <typename T>
class MusaIsNanOp : public MusaOpKernel {
 public:
  explicit MusaIsNanOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    // IsNan is elementwise; output shape is exactly the same as input.
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    const int64 num_elements = input.NumElements();
    if (num_elements == 0) return;

    OP_REQUIRES(
        ctx, num_elements <= std::numeric_limits<int>::max(),
        // The current launcher takes int element count.
        errors::InvalidArgument("IsNan input too large: ", num_elements));

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    IsNanLaunchHelper<T>::Run(input.flat<T>().data(), output->flat<bool>().data(),
                              static_cast<int>(num_elements), stream);
  }
};

#define REGISTER_MUSA_ISNAN(TYPE) \
  REGISTER_KERNEL_BUILDER(        \
      Name("IsNan").Device("MUSA").TypeConstraint<TYPE>("T"), MusaIsNanOp<TYPE>)

// Keep dtype coverage aligned with TF IsNan floating-point inputs.
REGISTER_MUSA_ISNAN(Eigen::half);
REGISTER_MUSA_ISNAN(float);
REGISTER_MUSA_ISNAN(double);

#undef REGISTER_MUSA_ISNAN

}  // namespace musa
}  // namespace tensorflow
