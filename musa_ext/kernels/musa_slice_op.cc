#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSliceOp : public MusaOpKernel {
 public:
  explicit MusaSliceOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Slice is memory-intensive but not computationally expensive
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& begin_tensor = ctx->input(1);
    const Tensor& size_tensor = ctx->input(2);

    const int dims = input.dims();
    std::vector<int64_t> starts_mt(dims);
    TensorShape output_shape;

    auto get_index_value = [](const Tensor& t, int i) -> int64_t {
      if (t.dtype() == DT_INT32) {
        return static_cast<int64_t>(t.flat<int32>()(i));
      } else {
        return static_cast<int64_t>(t.flat<int64_t>()(i));
      }
    };

    for (int i = 0; i < dims; ++i) {
      int64_t b = get_index_value(begin_tensor, i);
      int64_t s = get_index_value(size_tensor, i);

      if (s == -1) {
        s = input.dim_size(i) - b;
      }

      starts_mt[i] = b;
      output_shape.AddDim(s);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*output);

    ::musa::dnn::Permute op;

    auto status = op.ConfigDimStrideForSlice(out_mt, in_mt, starts_mt.data());
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN ConfigDimStrideForSlice failed. "
                                 "Check if input/output dims are consistent."));

    status = op.Run(handle, out_mt, in_mt);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Slice Run failed"));
  }
};

#define REGISTER_MUSA_SLICE(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_MTGPU)      \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          MusaSliceOp<type>);

REGISTER_MUSA_SLICE(float);        // FP32
REGISTER_MUSA_SLICE(double);       // FP64
REGISTER_MUSA_SLICE(int32);        // INT32
REGISTER_MUSA_SLICE(int64);        // INT64
REGISTER_MUSA_SLICE(Eigen::half);  // FP16
REGISTER_MUSA_SLICE(bfloat16);     // BF16

}  // namespace musa
}  // namespace tensorflow
