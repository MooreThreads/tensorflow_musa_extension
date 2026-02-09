#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

// 1. 注册 Op 定义 (复用 TF 标准定义，通常不需要重写 REGISTER_OP，但为了完整性)


template <typename T>
class MusaPlaceholderOp : public OpKernel {
 public:
  explicit MusaPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Placeholder 可能有一些 shape 属性，但通常不需要在构造函数里做重活
  }

  void Compute(OpKernelContext* ctx) override {
   
    
    if (ctx->output_required(0)) {
      ctx->CtxFailure(errors::InvalidArgument(
          "You must feed a value for placeholder tensor '", name(), 
          "' with dtype ", DataTypeString(output_type(0))));
    }
    
   
  }
};

// 2. 注册 Kernel

#define REGISTER_PLACEHOLDER(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("Placeholder")                       \
                              .Device("MUSA")                       \
                              .TypeConstraint<TYPE>("dtype"),       \
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
