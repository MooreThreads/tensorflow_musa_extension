#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaPlaceholderOp : public OpKernel {
 public:
  explicit MusaPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    if (ctx->output_required(0)) {
      ctx->CtxFailure(errors::InvalidArgument(
          "You must feed a value for placeholder tensor '", name(),
          "' with dtype ", DataTypeString(output_type(0))));
    }
  }
};

#define REGISTER_PLACEHOLDER(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Placeholder").Device("MUSA").TypeConstraint<TYPE>("dtype"), \
      MusaPlaceholderOp<TYPE>);

REGISTER_PLACEHOLDER(float);
REGISTER_PLACEHOLDER(double);
REGISTER_PLACEHOLDER(Eigen::half);
REGISTER_PLACEHOLDER(bfloat16);
REGISTER_PLACEHOLDER(int32);
REGISTER_PLACEHOLDER(int64);
REGISTER_PLACEHOLDER(bool);

#undef REGISTER_PLACEHOLDER

}  // namespace musa
}  // namespace tensorflow
