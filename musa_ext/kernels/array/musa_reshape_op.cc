#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaReshapeOp : public MusaOpKernel {
 public:
  explicit MusaReshapeOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& sizes = ctx->input(1);

    // 解析目标 shape
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

    // zero-copy core logic: construct a new Tensor that shares the underlying
    // GPU memory buffer, only replacing the shape view
    Tensor output;
    bool success = output.CopyFrom(input, shape);
    OP_REQUIRES(ctx, success,
                errors::Internal("MUSA Reshape: Tensor::CopyFrom failed."));

    ctx->set_output(0, output);
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
REGISTER_MUSA_RESHAPE(bool);

#undef REGISTER_MUSA_RESHAPE

}  // namespace musa
}  // namespace tensorflow
