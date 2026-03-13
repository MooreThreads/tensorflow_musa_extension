#include <limits>

#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
void LaunchTopKV2(const T* input, T* values, Tidx* indices, int rows, int cols,
                  int k, bool sorted, musaStream_t stream);

template <typename T>
class MusaTopKV2Op : public MusaOpKernel {
 public:
  explicit MusaTopKV2Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sorted", &sorted_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& k_tensor = ctx->input(1);

    OP_REQUIRES(
        ctx, input.dims() >= 1,
        errors::InvalidArgument(
            "TopKV2: input must be at least rank 1, got rank ", input.dims()));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(k_tensor.shape()),
        errors::InvalidArgument("TopKV2: k must be a scalar, got shape ",
                                k_tensor.shape().DebugString()));

    int64_t k64 = 0;
    switch (k_tensor.dtype()) {
      case DT_INT16:
        k64 = static_cast<int64_t>(k_tensor.scalar<int16>()());
        break;
      case DT_INT32:
        k64 = static_cast<int64_t>(k_tensor.scalar<int32>()());
        break;
      case DT_INT64:
        k64 = static_cast<int64_t>(k_tensor.scalar<int64>()());
        break;
      default:
        ctx->CtxFailure(
            errors::InvalidArgument("TopKV2: k must be int16/int32/int64, got ",
                                    DataTypeString(k_tensor.dtype())));
        return;
    }

    OP_REQUIRES(
        ctx, k64 >= 0,
        errors::InvalidArgument("TopKV2: k must be non-negative, got ", k64));

    const int64_t last_dim64 = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        ctx, k64 <= last_dim64,
        errors::InvalidArgument("TopKV2: k must not exceed last dim. k=", k64,
                                ", last_dim=", last_dim64));

    OP_REQUIRES(
        ctx,
        last_dim64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        errors::InvalidArgument("TopKV2: last dimension too large, cols=",
                                last_dim64));
    OP_REQUIRES(ctx,
                k64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                errors::InvalidArgument("TopKV2: k too large, k=", k64));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(input.dims() - 1, k64);

    Tensor* values = nullptr;
    Tensor* indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &values));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &indices));

    if (input.NumElements() == 0 || k64 == 0) {
      return;
    }

    const int cols = static_cast<int>(last_dim64);
    const int k = static_cast<int>(k64);
    OP_REQUIRES(
        ctx, k <= 1024,
        errors::InvalidArgument(
            "TopKV2: current MUSA implementation supports k <= 1024, got ", k));

    const int64_t rows64 = input.NumElements() / last_dim64;
    OP_REQUIRES(
        ctx, rows64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        errors::InvalidArgument("TopKV2: row count too large, rows=", rows64));
    const int rows = static_cast<int>(rows64);

    const T* input_ptr = input.flat<T>().data();
    T* values_ptr = values->flat<T>().data();
    int32* indices_ptr = indices->flat<int32>().data();

    auto* device = GetDeviceByCtx(ctx);
    auto stream = device->GetStream();

    LaunchTopKV2<T, int32>(input_ptr, values_ptr, indices_ptr, rows, cols, k,
                           sorted_, stream);
  }

 private:
  bool sorted_;
};

#define REGISTER_MUSA_TOPK(T)                          \
  REGISTER_KERNEL_BUILDER(Name("TopKV2")               \
                              .Device(DEVICE_MTGPU)    \
                              .HostMemory("k")         \
                              .TypeConstraint<T>("T"), \
                          MusaTopKV2Op<T>)

REGISTER_MUSA_TOPK(float);
REGISTER_MUSA_TOPK(Eigen::half);
REGISTER_MUSA_TOPK(bfloat16);

#undef REGISTER_MUSA_TOPK

}  // namespace musa
}  // namespace tensorflow