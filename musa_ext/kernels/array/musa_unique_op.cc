#include <list>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_memset.h"

namespace tensorflow {
namespace musa {
template <typename T, typename OutIdxT>
class MusaUniqueOp : public MusaOpKernel {
 public:
  explicit MusaUniqueOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() <= 1,
                errors::InvalidArgument("Unique only supports 1D tensor, got ",
                                        input.dims(), "D"));

    const int64_t input_num = input.NumElements();
    if (input_num == 0) {
      ctx->set_output(0, Tensor(input.dtype(), TensorShape()));
      ctx->set_output(1, Tensor(DataTypeToEnum<OutIdxT>::value, TensorShape()));
      return;
    }

    Tensor temp_out_values;
    Tensor* temp_out_indices;
    Tensor temp_counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                                           &temp_out_values));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, input.shape(), &temp_out_indices));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<OutIdxT>::value,
                                           input.shape(), &temp_counts));
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
    auto maintainer = MakeMusaMemMaintainer(mem_alloc_func);

    ::musa::dnn::Tensor t_in = CreateMTensor(input);
    ::musa::dnn::Tensor t_out_val = CreateMTensor(temp_out_values);
    ::musa::dnn::Tensor t_out_indices = CreateMTensor(*temp_out_indices);
    ::musa::dnn::Tensor t_counts = CreateMTensor(temp_counts);

    ::musa::dnn::Unique op;
    ::musa::dnn::Status mode_status =
        op.SetMode(::musa::dnn::Unique::Mode::UNSORTED);
    if (mode_status != ::musa::dnn::Status::SUCCESS) {
      ctx->SetStatus(errors::Internal("Unique SetMode failed"));
      return;
    }
    auto& handle = GetHandleByCtx(ctx);
    op.Run(handle, t_out_val, t_out_indices, t_counts, t_in, maintainer);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    std::vector<int64_t> nd_info;
    t_counts.GetNdInfo(nd_info);
    int64_t num_unique = nd_info[0];
    Tensor* out_values = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({num_unique}), &out_values));
    musaMemcpyAsync(out_values->data(), temp_out_values.data(),
                    num_unique * sizeof(T), musaMemcpyDeviceToDevice, stream);
  }
};

#define REGISTER_MUSA_UNIQUE(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_MTGPU)              \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          MusaUniqueOp<type, int32>);            \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_MTGPU)              \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          MusaUniqueOp<type, int64>);

REGISTER_MUSA_UNIQUE(float);
REGISTER_MUSA_UNIQUE(double);
REGISTER_MUSA_UNIQUE(int32);
REGISTER_MUSA_UNIQUE(int64);
REGISTER_MUSA_UNIQUE(Eigen::half);
REGISTER_MUSA_UNIQUE(bfloat16);

}  // namespace musa
}  // namespace tensorflow
