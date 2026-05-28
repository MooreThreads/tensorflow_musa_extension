#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

template <typename T>
void MusaDiagPartkernelLauncher(musaStream_t stream, uint64_t size, const T* in,
                                T* out);

template <typename T>
void MusaMatrixDiagPartV3KernelLauncher(musaStream_t stream, int64 batch_size,
                                         int64 M, int64 N, int k_min, int k_max,
                                         int64 num_diags, int64 max_diag_len,
                                         const T padding_value, const T* input,
                                         T* output);

template <typename T>
class MusaDiagPartOp : public MusaOpKernel {
  /*
    Implementation for DiagPart op, which extracts the diagonal part of a
  tensor.
    The shape of input should be like, considiering a tensor with a dim of 2k:
  [s1, s2, ..., sk, s1, s2, ..., sk]
    Then the output will be a tensor with shape [s1, s2, ..., sk]
  */
 public:
  explicit MusaDiagPartOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(
        context, 0 == num_dims % 2,
        errors::InvalidArgument("Input must have even number of dimensions"));

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      OP_REQUIRES(
          context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
          errors::InvalidArgument("Invalid shape ",
                                  tensor.shape().DebugString(), ": dimensions ",
                                  i, " and ", i + out_dims, " do not match."));
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(tensor.dim_size(i)));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Get stream
    MUSA_OP_REQUIRES_MUDNN_HANDLE(context);
    auto& handle = GetHandleByCtx(context);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    // Launch kernel
    MusaDiagPartkernelLauncher<T>(stream, output->NumElements(),
                                  tensor.flat<T>().data(),
                                  output->flat<T>().data());
  }
};  // class MusaDiagPartOp

#define REGISTER_MUSA_DIAG_PART(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DiagPart").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaDiagPartOp<TYPE>)

REGISTER_MUSA_DIAG_PART(float);
REGISTER_MUSA_DIAG_PART(double);
REGISTER_MUSA_DIAG_PART(int32);
REGISTER_MUSA_DIAG_PART(int64);
REGISTER_MUSA_DIAG_PART(Eigen::half);
REGISTER_MUSA_DIAG_PART(bfloat16);

// ==========================================
// MatrixDiagPartV3 Op
// ==========================================
// Handles tf.linalg.diag_part(x) which maps to MatrixDiagPartV3.
// Supports batched [..., M, N] inputs with scalar or range k.
template <typename T>
class MusaMatrixDiagPartV3Op : public MusaOpKernel {
 public:
  explicit MusaMatrixDiagPartV3Op(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& k_tensor = ctx->input(1);
    const Tensor& padding_tensor = ctx->input(2);

    // Parse k (scalar or 1-D length-2 vector)
    int k_min, k_max;
    const int32* k_data = k_tensor.flat<int32>().data();
    if (k_tensor.NumElements() == 1) {
      k_min = k_max = static_cast<int>(k_data[0]);
    } else {
      k_min = static_cast<int>(k_data[0]);
      k_max = static_cast<int>(k_data[1]);
    }
    OP_REQUIRES(ctx, k_min <= k_max,
                errors::InvalidArgument("k[0] must be <= k[1], got k=[",
                                        k_min, ",", k_max, "]"));

    const T padding_value = padding_tensor.scalar<T>()();
    const bool scalar_diag = (k_min == k_max);
    const int num_diags = k_max - k_min + 1;

    const TensorShape& in_shape = input.shape();
    const int ndims = in_shape.dims();
    OP_REQUIRES(ctx, ndims >= 2,
                errors::InvalidArgument("Input must be at least 2D, got rank ",
                                        ndims));

    const int64 M = in_shape.dim_size(ndims - 2);
    const int64 N = in_shape.dim_size(ndims - 1);

    // Compute max_diag_len across all requested diagonals
    int64 max_diag_len = 0;
    for (int k = k_min; k <= k_max; ++k) {
      int64 dl = std::min(M, N) - static_cast<int64>(std::abs(k));
      if (dl > max_diag_len) max_diag_len = dl;
    }
    OP_REQUIRES(ctx, max_diag_len > 0,
                errors::InvalidArgument("k is out of bounds for matrix [", M,
                                        ", ", N, "]"));

    // Build output shape: [...batch_dims..., (num_diags if range,) max_diag_len]
    TensorShape out_shape;
    for (int i = 0; i < ndims - 2; ++i) {
      out_shape.AddDim(in_shape.dim_size(i));
    }
    if (!scalar_diag) out_shape.AddDim(static_cast<int64>(num_diags));
    out_shape.AddDim(max_diag_len);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    const int64 batch_size = input.NumElements() / (M * N);

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    MusaMatrixDiagPartV3KernelLauncher<T>(
        stream, batch_size, M, N, k_min, k_max,
        static_cast<int64>(num_diags), max_diag_len, padding_value,
        input.flat<T>().data(), output->flat<T>().data());
  }
};

#define REGISTER_MUSA_MATRIX_DIAG_PART_V3(TYPE)                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MatrixDiagPartV3").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMatrixDiagPartV3Op<TYPE>)

REGISTER_MUSA_MATRIX_DIAG_PART_V3(float);
REGISTER_MUSA_MATRIX_DIAG_PART_V3(double);
REGISTER_MUSA_MATRIX_DIAG_PART_V3(int32);
REGISTER_MUSA_MATRIX_DIAG_PART_V3(int64);
REGISTER_MUSA_MATRIX_DIAG_PART_V3(Eigen::half);
REGISTER_MUSA_MATRIX_DIAG_PART_V3(bfloat16);

}  // namespace musa
}  // namespace tensorflow
