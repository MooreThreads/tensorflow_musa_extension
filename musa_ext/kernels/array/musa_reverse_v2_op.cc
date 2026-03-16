#include <cstdint>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

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
                  errors::InvalidArgument("axis[", i, "] = ", axis,
                                          " is out of valid range [", -dims,
                                          ", ", dims, ")."));
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

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    std::vector<int> axes_to_reverse;
    for (int i = 0; i < dims; ++i) {
      if (reverse_flags[i]) axes_to_reverse.push_back(i);
    }

    Tensor tmp;
    if (axes_to_reverse.size() > 1) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(input.dtype(), input.shape(), &tmp));
    }

    for (size_t idx = 0; idx < axes_to_reverse.size(); ++idx) {
      int ax = axes_to_reverse[idx];
      int64_t axis_size = input.dim_size(ax);

      const Tensor& src = (idx == 0) ? input : ((idx % 2 == 1) ? *output : tmp);
      Tensor* dst = (idx % 2 == 0) ? output : &tmp;

      Tensor idx_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({axis_size}),
                                             &idx_tensor));
      std::vector<int32> idx_host(axis_size);
      for (int64_t j = 0; j < axis_size; ++j) {
        idx_host[j] = static_cast<int32>(axis_size - 1 - j);
      }
      // Use async memcpy with pinned memory for better performance
      // The kernel launch will naturally wait for the memcpy to complete
      // as they are on the same stream
      musaMemcpyAsync(idx_tensor.data(), idx_host.data(),
                      axis_size * sizeof(int32), musaMemcpyHostToDevice,
                      stream);

      auto in_mt = CreateMTensor(src);
      auto out_mt = CreateMTensor(*dst);
      auto idx_mt = CreateMTensor(idx_tensor);

      mGatherX gather_op;
      gather_op.SetMode(mGatherX::Mode::GATHER);
      gather_op.SetAxis(ax);

      MTOP_CHECK_OK_RUN(gather_op.Run(handle, out_mt, idx_mt, in_mt),
                        "ReverseV2 GatherX Run", ctx);
    }

    if (axes_to_reverse.size() > 1 && axes_to_reverse.size() % 2 == 0) {
      musaMemcpyAsync(output->data(), tmp.data(), tmp.TotalBytes(),
                      musaMemcpyDeviceToDevice, stream);
    }
  }
};

#define REGISTER_MUSA_REVERSE_V2(T)                          \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                  \
                              .Device("MUSA")                \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<int32>("Tidx") \
                              .HostMemory("axis"),           \
                          MusaReverseV2Op<T, int32>);        \
  REGISTER_KERNEL_BUILDER(Name("ReverseV2")                  \
                              .Device("MUSA")                \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<int64>("Tidx") \
                              .HostMemory("axis"),           \
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
