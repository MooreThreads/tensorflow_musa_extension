#include <mudnn.h>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T, typename Tdim>
class MusaExpandDimsOp : public MusaOpKernel {
 public:
  explicit MusaExpandDimsOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dim_tensor = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dim_tensor.shape()),
                errors::InvalidArgument("dim input must be a scalar"));
    Tdim dim = dim_tensor.scalar<Tdim>()();
    const int input_dims = input.dims();

    if (dim < 0) {
      dim += input_dims + 1;
    }

    OP_REQUIRES(
        context, dim >= 0 && dim <= input_dims,
        errors::InvalidArgument("Inserted dimension ", dim,
                                " must be in range [0, ", input_dims, "]"));

    TensorShape out_shape;
    for (int i = 0; i < dim; ++i) {
      out_shape.AddDim(input.dim_size(i));
    }
    out_shape.AddDim(1);
    for (int i = dim; i < input_dims; ++i) {
      out_shape.AddDim(input.dim_size(i));
    }

    // OPTIMIZATION: Try buffer forwarding first
    // ExpandDims only adds a dimension of size 1, doesn't change data
    if (input.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      return;
    }

    // Try to use input buffer directly
    Tensor output_tensor(input.dtype());
    if (output_tensor.CopyFrom(input, out_shape)) {
      context->set_output(0, output_tensor);
      return;
    }

    // Fallback to forward_input_or_allocate_output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, out_shape, &output));

    // If forwarding didn't work, copy data
    if (output->tensor_data().data() != input.tensor_data().data()) {
      // GPU async copy for large tensors
      auto& h = GetHandleByCtx(context);
      musaStream_t stream = reinterpret_cast<musaStream_t>(h.GetStream());
      mStatus copy_status = MusaMemcpyAsyncD2D(
          const_cast<char*>(output->tensor_data().data()),
          input.tensor_data().data(), input.TotalBytes(), stream);
      OP_REQUIRES(context, copy_status == mStatus::SUCCESS,
                  errors::Internal("MUSA ExpandDims: async copy failed."));
    }
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
