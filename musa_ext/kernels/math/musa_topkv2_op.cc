#include <limits>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

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

    const int k = static_cast<int>(k64);
    const int dim = input.dims() - 1;

    auto& handle = GetHandleByCtx(ctx);
    auto* musa_device = static_cast<MusaDevice*>(ctx->device());

    std::vector<Tensor> workspace_tensors;
    auto mem_alloc_func =
        [ctx, &workspace_tensors](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return nullptr;
      Tensor temp;
      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return nullptr;
      workspace_tensors.emplace_back(temp);
      return ::musa::dnn::MemoryHandler(temp.flat<uint8_t>().data(),
                                        [](void*) {});
    };
    auto maintainer = musa_device->GetMemMaintainer(mem_alloc_func);

    mTensor input_mt = CreateMTensor(input);
    mTensor values_mt = CreateMTensor(*values);
    mTensor indices_mt = CreateMTensor(*indices);

    mTopK topk_op;
    MTOP_CHECK_OK(topk_op.SetK(k), "TopKV2 SetK", ctx);
    MTOP_CHECK_OK(topk_op.SetDim(dim), "TopKV2 SetDim", ctx);
    MTOP_CHECK_OK(topk_op.SetLargest(true), "TopKV2 SetLargest", ctx);
    MTOP_CHECK_OK(topk_op.SetSorted(sorted_), "TopKV2 SetSorted", ctx);
    MTOP_CHECK_OK(
        topk_op.Run(handle, values_mt, indices_mt, input_mt, maintainer),
        "TopKV2 Run", ctx);
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
REGISTER_MUSA_TOPK(int32);
REGISTER_MUSA_TOPK(Eigen::half);
REGISTER_MUSA_TOPK(bfloat16);
// Note: double and int64 are not supported by muDNN TopK.

#undef REGISTER_MUSA_TOPK

}  // namespace musa
}  // namespace tensorflow
