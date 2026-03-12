#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
class MusaReverseV2Op : public MusaOpKernel {
 public:
  explicit MusaReverseV2Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axis_tensor = ctx->input(1);
    const int dims = input.dims();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(axis_tensor.shape()),
                errors::InvalidArgument("axis must be 1-D, got shape ",
                                        axis_tensor.shape().DebugString()));

    const int64_t axis_num = axis_tensor.NumElements();
    auto axis_flat = axis_tensor.flat<Tidx>();

    std::vector<bool> reverse_flags(dims, false);
    bool need_reverse = false;

    for (int64_t i = 0; i < axis_num; ++i) {
      int64_t axis = static_cast<int64_t>(axis_flat(i));
      OP_REQUIRES(ctx, axis >= -dims && axis < dims,
                  errors::InvalidArgument(
                      "axis[", i, "] = ", axis, " is out of valid range [",
                      -dims, ", ", dims, ")."));
      if (axis < 0) axis += dims;
      OP_REQUIRES(
          ctx, !reverse_flags[axis],
          errors::InvalidArgument("axis ", axis, " is duplicated in axis."));
      reverse_flags[axis] = true;
      need_reverse = true;
    }

    if (!need_reverse || dims == 0 || input.NumElements() == 0) {
      ctx->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (output->NumElements() == 0) return;

    std::vector<int64_t> starts(dims, 0);
    std::vector<int64_t> strides(dims, 1);
    for (int i = 0; i < dims; ++i) {
      if (reverse_flags[i]) {
        starts[i] = input.dim_size(i) - 1;
        strides[i] = -1;
      }
    }

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*output);
    ::musa::dnn::Permute op;

    MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, starts.data(),
                                             strides.data()),
                  "ReverseV2 ConfigDimStrideForSlice", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt), "ReverseV2 Run", ctx);
  }
};

#define REGISTER_MUSA_REVERSE_V2(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                         \
                              .Device("MUSA")                       \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("axis"),                  \
                          MusaReverseV2Op<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                         \
                              .Device("MUSA")                       \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64>("Tidx")        \
                              .HostMemory("axis"),                  \
                          MusaReverseV2Op<T, int64>);

REGISTER_MUSA_REVERSE_V2(float);
REGISTER_MUSA_REVERSE_V2(double);
REGISTER_MUSA_REVERSE_V2(Eigen::half);
REGISTER_MUSA_REVERSE_V2(bfloat16);
REGISTER_MUSA_REVERSE_V2(int32);
REGISTER_MUSA_REVERSE_V2(int64);
REGISTER_MUSA_REVERSE_V2(bool);

#undef REGISTER_MUSA_REVERSE_V2

}  // namespace musa
}  // namespace tensorflow
