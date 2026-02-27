#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaReshapeOp : public MusaOpKernel {
 public:
  explicit MusaReshapeOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& sizes = ctx->input(1);

    TensorShape shape;
    int64 unknown_index = -1;
    int64 product = 1;

    if (sizes.dtype() == DT_INT32) {
      auto vec = sizes.flat<int32>();
      for (int i = 0; i < vec.size(); ++i) {
        int64 size = static_cast<int64>(vec(i));
        if (size == -1) {
          OP_REQUIRES(ctx, unknown_index == -1,
                      errors::InvalidArgument(
                          "Only one input size may be -1, not both ",
                          unknown_index, " and ", i));
          unknown_index = i;
          shape.AddDim(1);
        } else {
          OP_REQUIRES(ctx, size >= 0,
                      errors::InvalidArgument(
                          "Dimension size must be non-negative, got ", size));
          shape.AddDim(size);
          product *= size;
        }
      }
    } else if (sizes.dtype() == DT_INT64) {
      auto vec = sizes.flat<int64>();
      for (int i = 0; i < vec.size(); ++i) {
        int64 size = vec(i);
        if (size == -1) {
          OP_REQUIRES(ctx, unknown_index == -1,
                      errors::InvalidArgument(
                          "Only one input size may be -1, not both ",
                          unknown_index, " and ", i));
          unknown_index = i;
          shape.AddDim(1);
        } else {
          OP_REQUIRES(ctx, size >= 0,
                      errors::InvalidArgument(
                          "Dimension size must be non-negative, got ", size));
          shape.AddDim(size);
          product *= size;
        }
      }
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("Shape tensor must be int32 or int64"));
    }

    if (unknown_index != -1) {
      int64 input_num_elements = input.NumElements();
      OP_REQUIRES(ctx, product > 0,
                  errors::InvalidArgument(
                      "Cannot infer -1 dimension with zero product"));
      OP_REQUIRES(ctx, input_num_elements % product == 0,
                  errors::InvalidArgument(
                      "Input has ", input_num_elements,
                      " elements, which isn't divisible by ", product));
      int64 inferred_dim = input_num_elements / product;
      shape.set_dim(unknown_index, inferred_dim);
    }

    OP_REQUIRES(ctx, input.NumElements() == shape.num_elements(),
                errors::InvalidArgument("Input has ", input.NumElements(),
                                        " elements, but target shape has ",
                                        shape.num_elements(), " elements."));

    // OPTIMIZATION: Try buffer forwarding first to avoid copy
    // Reshape doesn't change data, only metadata
    if (input.NumElements() > 0) {
      // Try to forward the input buffer to output
      // This avoids memory copy when possible
      Tensor* output = nullptr;

      // Check if we can use set_output (same underlying buffer)
      if (shape.num_elements() == input.NumElements()) {
        // Use the input tensor's buffer directly
        Tensor output_tensor(input.dtype());
        if (output_tensor.CopyFrom(input, shape)) {
          ctx->set_output(0, output_tensor);
          return;
        }
      }

      // Fallback to forward_input_or_allocate_output
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output({0}, 0, shape, &output));

      if (output->NumElements() == 0) return;

      // If forwarding didn't work (output is new allocation), copy data
      if (output->tensor_data().data() != input.tensor_data().data()) {
        auto& handle = GetHandleByCtx(ctx);
        musaStream_t stream =
            reinterpret_cast<musaStream_t>(handle.GetStream());

        mStatus status = MusaMemcpyAsyncD2D(
            const_cast<char*>(output->tensor_data().data()),
            input.tensor_data().data(), input.TotalBytes(), stream);

        OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                    errors::Internal("MUSA Reshape: async copy failed"));
      }
    } else {
      // Empty tensor - just allocate output
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    }
  }
};

#define REGISTER_MUSA_RESHAPE(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Reshape").Device("MUSA").TypeConstraint<TYPE>("T").HostMemory( \
          "shape"),                                                        \
      MusaReshapeOp<TYPE>)

REGISTER_MUSA_RESHAPE(float);
REGISTER_MUSA_RESHAPE(Eigen::half);
REGISTER_MUSA_RESHAPE(bfloat16);
REGISTER_MUSA_RESHAPE(double);
REGISTER_MUSA_RESHAPE(int32);
REGISTER_MUSA_RESHAPE(int64);

#undef REGISTER_MUSA_RESHAPE

}  // namespace musa
}  // namespace tensorflow
