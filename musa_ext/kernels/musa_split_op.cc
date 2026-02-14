#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSplitOp : public OpKernel {
 public:
  explicit MusaSplitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& split_dim_tensor = context->input(0);
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32 num_split = context->num_outputs();

    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(split_dim_tensor.shape()),
        errors::InvalidArgument("split_dim must be a scalar, but got rank ",
                                split_dim_tensor.shape().dims()));

    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim =
        split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    OP_REQUIRES(
        context, 0 <= split_dim && split_dim < input.dims(),
        errors::InvalidArgument("Split dim ", split_dim_orig, " out of range"));

    OP_REQUIRES(context, num_split > 0,
                errors::InvalidArgument("Number of ways to split must be > 0"));

    const int64_t input_size_split_dim = input_shape.dim_size(split_dim);
    OP_REQUIRES(
        context, input_size_split_dim % num_split == 0,
        errors::InvalidArgument(
            "Number of ways to split must evenly divide the split dimension"));

    if (num_split == 1) {
      context->set_output(0, input);
      return;
    }

    const int64_t delta = input_size_split_dim / num_split;
    auto& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;

    std::vector<int64_t> starts_mt(input.dims(), 0);
    std::vector<int64_t> strides_mt(input.dims(), 1);

    TensorShape out_shape = input_shape;
    out_shape.set_dim(split_dim, delta);

    for (int i = 0; i < num_split; ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(i, out_shape, &output));

      if (delta == 0) continue;

      auto in_mt = CreateMTensor(input);
      auto out_mt = CreateMTensor(*output);

      starts_mt[split_dim] = i * delta;

      MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, starts_mt.data(),
                                               strides_mt.data()),
                    "ConfigDimStrideForSlice", context);

      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Split Run", context);
    }
  }
};

#define REGISTER_MUSA_SPLIT(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Split").Device("MUSA").TypeConstraint<type>("T").HostMemory( \
          "split_dim"),                                                  \
      MusaSplitOp<type>)

REGISTER_MUSA_SPLIT(float);
REGISTER_MUSA_SPLIT(double);
REGISTER_MUSA_SPLIT(Eigen::half);
REGISTER_MUSA_SPLIT(Eigen::bfloat16);
REGISTER_MUSA_SPLIT(int32);
REGISTER_MUSA_SPLIT(int64);
REGISTER_MUSA_SPLIT(bool);
REGISTER_MUSA_SPLIT(uint8);

#undef REGISTER_MUSA_SPLIT

}  // namespace musa
}  // namespace tensorflow
