#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T, typename Tdim>
class MusaExpandDimsOp : public MusaOpKernel {
 public:
  explicit MusaExpandDimsOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  // ExpandDims is a pure metadata operation (zero copy), mark as inexpensive
  // to enable TensorFlow executor optimizations (e.g., inline scheduling).
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    // Reject Variant type early to align with TensorFlow official behavior.
    OP_REQUIRES(context, context->input(0).dtype() != DT_VARIANT,
                errors::InvalidArgument(
                    "ExpandDims on Variant type is not supported"));
    const Tensor& input = context->input(0);
    const Tensor& dim_tensor = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dim_tensor.shape()),
                errors::InvalidArgument("dim input must be a scalar"));

    Tdim dim = dim_tensor.scalar<Tdim>()();
    const int input_dims = input.dims();

    // Normalize negative dimension
    if (dim < 0) {
      dim += input_dims + 1;
    }

    OP_REQUIRES(
        context, dim >= 0 && dim <= input_dims,
        errors::InvalidArgument("Inserted dimension ", dim,
                                " must be in range [0, ", input_dims, "]"));

    // Build output shape: insert dimension of size 1 at position 'dim'
    TensorShape out_shape;
    out_shape.AppendShape(input.shape());
    out_shape.InsertDim(dim, 1);

    // Handle empty tensor
    if (input.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      return;
    }

    // Use CopyFrom for zero-copy buffer sharing.
    // ExpandDims only changes metadata (adds a dimension of size 1),
    // the underlying data buffer remains exactly the same.
    // CopyFrom will forward the buffer if possible (ref count == 1),
    // otherwise it will allocate and copy the data.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    OP_REQUIRES(
        context, output->CopyFrom(input, out_shape),
        errors::Internal("Could not expand dimension: shape mismatch. Input shape: ",
                         input.shape().DebugString(), ", output shape: ",
                         out_shape.DebugString()));
  }
};

#define REGISTER_MUSA_EXPAND_DIMS(type)                      \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device("MUSA")                \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          MusaExpandDimsOp<type, int32>);    \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device("MUSA")                \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int64>("Tdim") \
                              .HostMemory("dim"),            \
                          MusaExpandDimsOp<type, int64>);

REGISTER_MUSA_EXPAND_DIMS(float);
REGISTER_MUSA_EXPAND_DIMS(int32);
REGISTER_MUSA_EXPAND_DIMS(int64);
REGISTER_MUSA_EXPAND_DIMS(Eigen::half);
REGISTER_MUSA_EXPAND_DIMS(bool);
REGISTER_MUSA_EXPAND_DIMS(double);
REGISTER_MUSA_EXPAND_DIMS(bfloat16);
REGISTER_MUSA_EXPAND_DIMS(uint8);

#undef REGISTER_MUSA_EXPAND_DIMS

}  // namespace
}  // namespace musa
}  // namespace tensorflow
