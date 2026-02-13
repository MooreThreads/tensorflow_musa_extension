#include <list>
#include <vector>

#include "mu/device/musa_device.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_memset.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename OutIdxT>
class MusaUniqueOp : public MusaOpKernel {
 public:
  explicit MusaUniqueOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() <= 1,
                errors::InvalidArgument("Unique expects a 1D vector."));

    const int64_t num_elements = input.NumElements();

    Tensor temp_out_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                                           &temp_out_values));

    Tensor* out_indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &out_indices));

    Tensor tmp_counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<OutIdxT>::value,
                                           input.shape(), &tmp_counts));

    if (num_elements == 0) {
      Tensor* empty_out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &empty_out));
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    auto* musa_device = static_cast<MusaDevice*>(ctx->device());

    size_t counts_bytes = num_elements * sizeof(OutIdxT);
    musaMemset(tmp_counts.flat<OutIdxT>().data(), 0, counts_bytes);

    std::list<Tensor> workspace_tensors;
    auto mem_alloc_func =
        [ctx, &workspace_tensors](size_t size) -> ::musa::dnn::MemoryHandler {
      workspace_tensors.emplace_back();
      Tensor& temp = workspace_tensors.back();

      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return nullptr;

      void* raw_ptr = static_cast<void*>(temp.flat<uint8_t>().data());
      return ::musa::dnn::MemoryHandler(raw_ptr, [](void* p) {});
    };

    ::musa::dnn::MemoryMaintainer maintainer =
        musa_device->GetMemMaintainer(mem_alloc_func);

    mTensor t_in = CreateMTensor(input);
    mTensor t_temp_out = CreateMTensor(temp_out_values);
    mTensor t_idx = CreateMTensor(*out_indices);
    mTensor t_counts = CreateMTensor(tmp_counts);

    ::musa::dnn::Unique op;
    auto status = op.SetMode(::musa::dnn::Unique::Mode::UNSORTED);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Unique SetMode failed"));

    status = op.Run(handle, t_temp_out, t_idx, t_counts, t_in, maintainer);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA muDNN Unique execution failed. Status: ",
                                 (int)status));

    std::vector<OutIdxT> host_counts(num_elements);
    musaMemcpy(host_counts.data(), tmp_counts.flat<OutIdxT>().data(),
               counts_bytes, musaMemcpyDeviceToHost);

    int64_t unique_count = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
      if (host_counts[i] > 0) {
        unique_count++;
      } else {
        break;
      }
    }

    Tensor* out_values = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({unique_count}), &out_values));

    if (unique_count > 0) {
      size_t data_bytes = unique_count * sizeof(T);
      musaMemcpy(out_values->flat<T>().data(), temp_out_values.flat<T>().data(),
                 data_bytes, musaMemcpyDeviceToDevice);
    }
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
