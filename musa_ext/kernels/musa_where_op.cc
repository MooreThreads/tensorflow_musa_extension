#include <mudnn.h>

#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaWhereOp : public MusaOpKernel {
 public:
  explicit MusaWhereOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    int64_t rank = input.dims();

    if (input.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({0, rank}), &output));
      return;
    }

    MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(ctx->device());
    auto& h = musa_device->mudnn_handle();

    auto input_mt = CreateMTensor(input);

    size_t captured_size = 0;
    void* scratch_ptr = nullptr;
    Tensor scratch_tensor_holder;

    auto mm = musa_device->GetMemMaintainer(
        [ctx, &captured_size, &scratch_ptr,
         &scratch_tensor_holder](size_t size) -> ::musa::dnn::MemoryHandler {
          captured_size = size;
          Status s = ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64_t>(size)}),
              &scratch_tensor_holder);
          if (!s.ok()) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});

          scratch_ptr =
              const_cast<char*>(scratch_tensor_holder.tensor_data().data());
          return ::musa::dnn::MemoryHandler(scratch_ptr, [](void*) {});
        });

    ::musa::dnn::Nonzero op;
    ::musa::dnn::Tensor out_mt;
    out_mt.SetType(mType::INT64);

    auto status = op.Run(h, out_mt, input_mt, mm);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Nonzero run failed"));

    musaStreamSynchronize(h.GetStream());

    if (captured_size == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({0, rank}), &output));
      return;
    }

    size_t element_size = sizeof(int64_t);
    size_t num_nonzero = captured_size / (rank * element_size);

    TensorShape output_shape({static_cast<int64_t>(num_nonzero), rank});
    Tensor* final_output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &final_output));

    auto dst_ptr = final_output->tensor_data().data();
    musaMemcpyAsync((void*)dst_ptr, scratch_ptr, captured_size,
                    musaMemcpyDeviceToDevice, h.GetStream());
  }
};

#define REGISTER_WHERE(T)                                        \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Where").Device(DEVICE_MTGPU).TypeConstraint<T>("T"), \
      MusaWhereOp<T>);

REGISTER_WHERE(float);
REGISTER_WHERE(double);
REGISTER_WHERE(int32);
REGISTER_WHERE(int64);
REGISTER_WHERE(bool);
REGISTER_WHERE(Eigen::half);
REGISTER_WHERE(Eigen::bfloat16);

#undef REGISTER_WHERE

}  // namespace musa
}  // namespace tensorflow