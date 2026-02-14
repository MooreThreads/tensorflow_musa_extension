#include <musa_runtime_api.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace musa {

template <typename OutType>
class MusaShapeOp : public OpKernel {
 public:
  explicit MusaShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& shape = input.shape();
    const int rank = shape.dims();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &output));

    auto flat_output = output->flat<OutType>();
    for (int i = 0; i < rank; ++i) {
      flat_output(i) = static_cast<OutType>(shape.dim_size(i));
    }
  }
};

template <typename OutType>
class MusaShapeNOp : public OpKernel {
 public:
  explicit MusaShapeNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& inp = ctx->input(i);
      const TensorShape& shape = inp.shape();
      const int rank = shape.dims();

      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, TensorShape({rank}), &output));

      auto flat_output = output->flat<OutType>();
      for (int j = 0; j < rank; ++j) {
        flat_output(j) = static_cast<OutType>(shape.dim_size(j));
      }
    }
  }
};

#define MUSA_SHAPE_INPUT_TYPES                                            \
  {                                                                       \
    DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_INT16, DT_INT8, DT_UINT8, \
        DT_HALF, DT_BFLOAT16, DT_BOOL                                     \
  }

#define REGISTER_MUSA_SHAPE_OPS(out_type)                                   \
                                                                            \
  REGISTER_KERNEL_BUILDER(Name("Shape")                                     \
                              .Device("MUSA")                               \
                              .HostMemory("output")                         \
                              .TypeConstraint<out_type>("out_type")         \
                              .TypeConstraint("T", MUSA_SHAPE_INPUT_TYPES), \
                          MusaShapeOp<out_type>);                           \
                                                                            \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                                    \
                              .Device("MUSA")                               \
                              .HostMemory("output")                         \
                              .TypeConstraint<out_type>("out_type")         \
                              .TypeConstraint("T", MUSA_SHAPE_INPUT_TYPES), \
                          MusaShapeNOp<out_type>)

REGISTER_MUSA_SHAPE_OPS(int32);
REGISTER_MUSA_SHAPE_OPS(int64);
REGISTER_MUSA_SHAPE_OPS(float);
REGISTER_MUSA_SHAPE_OPS(Eigen::half);
REGISTER_MUSA_SHAPE_OPS(bfloat16);
REGISTER_MUSA_SHAPE_OPS(double);

#undef REGISTER_MUSA_SHAPE_OPS
#undef MUSA_SHAPE_INPUT_TYPES

}  // namespace musa
}  // namespace tensorflow
