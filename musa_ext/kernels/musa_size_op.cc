#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSizeOp : public MusaOpKernel {
 public:
  explicit MusaSizeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    const int64_t num_elements = input.NumElements();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    if (output->dtype() == DT_INT32) {
      OP_REQUIRES(
          ctx, num_elements <= std::numeric_limits<int32>::max(),
          errors::InvalidArgument("Number of elements exceeds int32 max"));
      output->scalar<int32>()() = static_cast<int32>(num_elements);
    } else if (output->dtype() == DT_INT64) {
      output->scalar<int64>()() = num_elements;
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("Size output must be int32 or int64"));
    }
  }
};

#define REGISTER_MUSA_SIZE(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device("MUSA")                    \
                              .TypeConstraint<TYPE>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          MusaSizeOp<TYPE>);                     \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device("MUSA")                    \
                              .TypeConstraint<TYPE>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          MusaSizeOp<TYPE>);

REGISTER_MUSA_SIZE(float);
REGISTER_MUSA_SIZE(double);
REGISTER_MUSA_SIZE(Eigen::half);
REGISTER_MUSA_SIZE(bfloat16);
REGISTER_MUSA_SIZE(int32);
REGISTER_MUSA_SIZE(int64);
REGISTER_MUSA_SIZE(bool);

#undef REGISTER_MUSA_SIZE

}  // namespace musa
}  // namespace tensorflow
