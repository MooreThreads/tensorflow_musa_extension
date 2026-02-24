#include <mudnn.h>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
class MusaArgMaxOp : public MusaOpKernel {
 public:
  explicit MusaArgMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axis_tensor = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be scalar"));

    int64_t axis = 0;
    if (axis_tensor.dtype() == DT_INT32) {
      axis = axis_tensor.scalar<int32>()();
    } else {
      axis = axis_tensor.scalar<int64>()();
    }

    const int input_dims = input.dims();
    OP_REQUIRES(ctx, axis >= -input_dims && axis < input_dims,
                errors::InvalidArgument("Expected axis in range [", -input_dims,
                                        ", ", input_dims, "), but got ", axis));
    if (axis < 0) axis += input_dims;

    const int64_t axis_size = input.dim_size(axis);
    OP_REQUIRES(ctx, axis_size > 0,
                errors::InvalidArgument(
                    "Reduction axis ", axis,
                    " is empty in shape: ", input.shape().DebugString()));

    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      if (i != axis) output_shape.AddDim(input.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() == 0) return;

    // 使用 muDNN TopK 实现 ArgMax
    auto& handle = GetHandleByCtx(ctx);
    
    // 创建输入和输出的 mTensor
    mTensor input_mt = CreateMTensor(input, format_);
    mTensor output_mt = CreateMTensor(*output, format_);
    
    // muDNN TopK 需要额外的值输出，但我们只需要索引
    Tensor temp_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), output_shape, &temp_values));
    mTensor values_mt = CreateMTensor(temp_values, format_);
    
    // 配置 TopK 算子
    ::musa::dnn::TopK topk_op;
    topk_op.SetK(1);  // 只需要最大值的索引
    topk_op.SetAxis(static_cast<int>(axis));
    topk_op.SetLargest(true);
    
    // 执行 TopK 操作
    auto status = topk_op.Run(handle, values_mt, output_mt, input_mt);
    
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA ArgMax execution failed. Status: ", static_cast<int>(status)));
  }
};

#define REGISTER_MUSA_ARGMAX(T, Tidx)                              \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                           \
                              .Device("MUSA")                      \
                              .TypeConstraint<T>("T")              \
                              .TypeConstraint<Tidx>("output_type") \
                              .HostMemory("dimension"),            \
                          MusaArgMaxOp<T, Tidx>);

REGISTER_MUSA_ARGMAX(float, int64);
REGISTER_MUSA_ARGMAX(float, int32);

REGISTER_MUSA_ARGMAX(int32, int64);
REGISTER_MUSA_ARGMAX(int32, int32);

REGISTER_MUSA_ARGMAX(Eigen::half, int64);
REGISTER_MUSA_ARGMAX(Eigen::half, int32);

REGISTER_MUSA_ARGMAX(bfloat16, int64);
REGISTER_MUSA_ARGMAX(bfloat16, int32);

REGISTER_MUSA_ARGMAX(double, int64);
REGISTER_MUSA_ARGMAX(double, int32);

#undef REGISTER_MUSA_ARGMAX

}  // namespace musa
}  // namespace tensorflow
