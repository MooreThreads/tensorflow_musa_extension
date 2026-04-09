#include "../utils_op.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

template <typename Scalar>
void MusaMatrixSetDiagKernelLauncher(musaStream_t,
                                     typename TTypes<Scalar, 3>::ConstTensor&,
                                     typename TTypes<Scalar, 3>::ConstTensor&,
                                     typename TTypes<Scalar, 3>::Tensor&,
                                     const Eigen::Index, const Eigen::Index,
                                     const Eigen::Index, const bool,
                                     const bool);

void ReadAlignment(OpKernelConstruction* context,
                   bool* left_align_superdiagonal,
                   bool* left_align_subdiagonal) {
  std::string align;
  OP_REQUIRES_OK(context, context->GetAttr("align", &align));

  *left_align_superdiagonal = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
  *left_align_subdiagonal = align == "RIGHT_LEFT" || align == "RIGHT_RIGHT";
}

template <typename T>
class MusaMatrixSetDiagOp : public MusaOpKernel {
 public:
  explicit MusaMatrixSetDiagOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {
    // MatrixSetDiag-specific
    if (context->HasAttr("align")) {
      ReadAlignment(context, &left_align_superdiagonal_,
                    &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
    MUSA_DEBUG_LOG_KERNEL(context);
    const Tensor& input = context->input(0);
    const Tensor& diag = context->input(1);

    int32_t lower_diag_index = 0;
    int32_t upper_diag_index = 0;

    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(2);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      OP_REQUIRES(
          context, diag_index.NumElements() > 0,
          errors::InvalidArgument("diag_index must have at least one element"));

      // Fetch lower & upper diag indices from diag_index input. If diag_index
      // is a scalar, both lower and upper diag indices are the same.
      lower_diag_index = diag_index.flat<int32_t>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            context, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32_t>()(1);
        }
      }
    }

    const TensorShape& input_shape = input.shape();
    const TensorShape& diag_shape = diag.shape();
    const int input_rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument(
                    "input must be at least 1-dim, received shape: ",
                    input_shape.DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const Eigen::Index num_rows = input_shape.dim_size(input_rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(input_rank - 1);
    OP_REQUIRES(context,
                (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
                    lower_diag_index == 0,
                errors::InvalidArgument(
                    "lower_diag_index is out of bound: ", lower_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    // Check if diag size is consistent with input
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;

    // Before accessing diag_shape.dim_size(input_rank - 2), we must ensure
    // the diagnoal tensor actually has that dimension. Accessing a dimension
    // index >= rank causes a hard crash (core dump).
    if (lower_diag_index != upper_diag_index) {
      OP_REQUIRES(
          context, diag_shape.dims() > input_rank - 2,
          errors::InvalidArgument(
              "Diagnoal tensor rank must be large enough to contain a "
              "diagnoal-count dimension. Input rank: ",
              input_rank, ", Expected diagnoal rank >= ", input_rank - 1,
              ", Received diagnoal rank: ", diag_shape.dims()));
    }

    OP_REQUIRES(
        context,
        lower_diag_index == upper_diag_index ||
            (diag_shape.dim_size(input_rank - 2) == num_diags),
        errors::InvalidArgument(
            "The number of diagnoals provided in `diag` is not consistent "
            "with `lower_diag_index` and `upper_diag_index`"));

    TensorShape expected_diag_shape = input_shape;
    expected_diag_shape.RemoveLastDims(2);
    if (num_diags > 1) {
      OP_REQUIRES_OK(context, expected_diag_shape.AddDimWithStatus(num_diags));
    }
    const int32_t max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    OP_REQUIRES_OK(context, expected_diag_shape.AddDimWithStatus(max_diag_len));
    OP_REQUIRES(
        context, expected_diag_shape == diag_shape,
        errors::InvalidArgument(
            "Either first dimensions of diagnoal don't match input.shape[:-2], "
            "or diagnoal.shape[:-1] is not equal to the longests diagnoal in "
            "range [lower_diag_index:upper_diag_index].\nInput shape: ",
            input_shape.DebugString(),
            "\nDiagnoal shape: ", diag_shape.DebugString(),
            "\nExpected diagnoal shape: ", expected_diag_shape.DebugString()));

    if (input.NumElements() == 0) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    auto input_reshaped = input.flat_inner_dims<T, 3>();
    auto diag_reshaped = diag.flat_inner_dims<T, 3>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();

    musaStream_t stream = GetMusaStreamByCtx(context);
    MusaMatrixSetDiagKernelLauncher<T>(
        stream, input_reshaped, diag_reshaped, output_reshaped,
        lower_diag_index, upper_diag_index, num_diags,
        left_align_superdiagonal_, left_align_subdiagonal_);
  }

  bool IsExpensive() override { return true; }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 2;
};

#define REGISTER_MATRIX_SET_DIAG_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MusaMatrixSetDiag").Device("MUSA").TypeConstraint<type>("T"),   \
      MusaMatrixSetDiagOp<type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MusaMatrixSetDiagV2").Device("MUSA").TypeConstraint<type>("T"), \
      MusaMatrixSetDiagOp<type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MusaMatrixSetDiagV3").Device("MUSA").TypeConstraint<type>("T"), \
      MusaMatrixSetDiagOp<type>);

REGISTER_MATRIX_SET_DIAG_KERNEL(float);
REGISTER_MATRIX_SET_DIAG_KERNEL(double);
REGISTER_MATRIX_SET_DIAG_KERNEL(int32_t);
REGISTER_MATRIX_SET_DIAG_KERNEL(int64_t);
REGISTER_MATRIX_SET_DIAG_KERNEL(Eigen::half);
REGISTER_MATRIX_SET_DIAG_KERNEL(Eigen::bfloat16);
REGISTER_MATRIX_SET_DIAG_KERNEL(bool);

#undef REGISTER_MATRIX_SET_DIAG_KERNEL
}  // namespace musa
}  // namespace tensorflow