#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace musa {

template <typename T>
void MusaMatrixBandPartKernelLauncher(musaStream_t stream, const int batch_size,
                                      const int m, const int n,
                                      const int num_lower_diags,
                                      const int num_upper_diags,
                                      const T* input_ptr, T* output_ptr);

template <typename T>
class MusaMatrixBandPartOp : public MusaOpKernel {
 public:
  explicit MusaMatrixBandPartOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));

    // Read num_lower and num_upper as int64 scalars (may be DT_INT32 or
    // DT_INT64)
    auto as_int64_scalar = [](const Tensor& t) -> int64 {
      if (t.dtype() == DT_INT32) {
        return static_cast<int64>(t.scalar<int32>()());
      } else {
        return t.scalar<int64>()();
      }
    };

    const Tensor& num_lower_in = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be a scalar, got: ",
                                        num_lower_in.shape().DebugString()));
    const int64 num_lower = as_int64_scalar(num_lower_in);

    const Tensor& num_upper_in = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be a scalar, got: ",
                                        num_upper_in.shape().DebugString()));
    const int64 num_upper = as_int64_scalar(num_upper_in);

    // Reshape to [..., m, n] -> [batch, m, n]
    auto input_reshaped = input.flat_inner_dims<T, 3>();
    const int64 batch_size = input_reshaped.dimension(0);
    const int64 m = input_reshaped.dimension(1);
    const int64 n = input_reshaped.dimension(2);

    OP_REQUIRES(ctx, num_lower <= m,
                errors::InvalidArgument(
                    "num_lower must be negative or <= number of rows (", m,
                    "), got: ", num_lower));
    OP_REQUIRES(ctx, num_upper <= n,
                errors::InvalidArgument(
                    "num_upper must be negative or <= number of columns (", n,
                    "), got: ", num_upper));

    // Passthrough if the entire matrix is within the band
    if (input.NumElements() == 0 || ((num_lower < 0 || num_lower >= m) &&
                                     (num_upper < 0 || num_upper >= n))) {
      ctx->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, input_shape, &output));

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = static_cast<musaStream_t>(handle.GetStream());

    MusaMatrixBandPartKernelLauncher<T>(
        stream, static_cast<int>(batch_size), static_cast<int>(m),
        static_cast<int>(n), static_cast<int>(num_lower),
        static_cast<int>(num_upper), input.flat<T>().data(),
        output->flat<T>().data());
  }
};

#define REGISTER_MUSA_MATRIX_BAND_PART(TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("MatrixBandPart")         \
                              .Device(DEVICE_MTGPU)      \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MusaMatrixBandPartOp<TYPE>)

REGISTER_MUSA_MATRIX_BAND_PART(float);
REGISTER_MUSA_MATRIX_BAND_PART(double);
REGISTER_MUSA_MATRIX_BAND_PART(int32);
REGISTER_MUSA_MATRIX_BAND_PART(int64);
REGISTER_MUSA_MATRIX_BAND_PART(Eigen::half);
REGISTER_MUSA_MATRIX_BAND_PART(bfloat16);

#undef REGISTER_MUSA_MATRIX_BAND_PART

}  // namespace musa
}  // namespace tensorflow
