#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace se = ::stream_executor;

template <typename T>
class MusaPackOp : public MusaOpKernel {
 public:
  explicit MusaPackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const int num = num_inputs();
    const Tensor& first_input = ctx->input(0);

    int expanded_num_dims = first_input.dims() + 1;
    int axis = axis_ < 0 ? axis_ + expanded_num_dims : axis_;

    TensorShape output_shape(first_input.shape());
    output_shape.InsertDim(axis, num);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    se::Stream* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("No MUSA stream available."));

    int64_t before_size = 1;
    for (int i = 0; i < axis; ++i) before_size *= output_shape.dim_size(i);

    int64_t after_size = 1;
    for (int i = axis + 1; i < output_shape.dims(); ++i)
      after_size *= output_shape.dim_size(i);

    const size_t copy_bytes = after_size * sizeof(T);
    T* out_base_ptr = const_cast<T*>(output->flat<T>().data());

    for (int i = 0; i < num; ++i) {
      const T* in_base_ptr = ctx->input(i).flat<T>().data();

      for (int b = 0; b < before_size; ++b) {
        int64_t out_offset = (b * num + i) * after_size;
        int64_t in_offset = b * after_size;

        se::DeviceMemoryBase out_mem(out_base_ptr + out_offset, copy_bytes);
        se::DeviceMemoryBase in_mem(const_cast<T*>(in_base_ptr + in_offset),
                                    copy_bytes);

        stream->ThenMemcpy(&out_mem, in_mem, copy_bytes);
      }
    }
  }

 private:
  int axis_;
};

template <typename T>
class MusaUnpackOp : public MusaOpKernel {
 public:
  explicit MusaUnpackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    int axis = axis_ < 0 ? axis_ + input.dims() : axis_;
    const int num = input.dim_size(axis);

    TensorShape output_shape = input.shape();
    output_shape.RemoveDim(axis);

    se::Stream* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("No MUSA stream available."));

    int64_t before_size = 1;
    for (int i = 0; i < axis; ++i) before_size *= input.dim_size(i);
    int64_t after_size = 1;
    for (int i = axis + 1; i < input.dims(); ++i)
      after_size *= input.dim_size(i);

    const size_t copy_bytes = after_size * sizeof(T);
    const T* in_base_ptr = input.flat<T>().data();

    for (int i = 0; i < num; ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output));
      if (output->NumElements() == 0) continue;

      T* out_base_ptr = const_cast<T*>(output->flat<T>().data());

      for (int b = 0; b < before_size; ++b) {
        int64_t in_offset = (b * num + i) * after_size;
        int64_t out_offset = b * after_size;

        se::DeviceMemoryBase out_mem(out_base_ptr + out_offset, copy_bytes);
        se::DeviceMemoryBase in_mem(const_cast<T*>(in_base_ptr + in_offset),
                                    copy_bytes);

        stream->ThenMemcpy(&out_mem, in_mem, copy_bytes);
      }
    }
  }

 private:
  int axis_;
};

#define REGISTER_MUSA_STACK_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Pack").Device(DEVICE_MTGPU).TypeConstraint<type>("T"),   \
      MusaPackOp<type>);                                             \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Unpack").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaUnpackOp<type>);

REGISTER_MUSA_STACK_KERNELS(float);
REGISTER_MUSA_STACK_KERNELS(double);
REGISTER_MUSA_STACK_KERNELS(int32);
REGISTER_MUSA_STACK_KERNELS(int64);
REGISTER_MUSA_STACK_KERNELS(Eigen::half);
REGISTER_MUSA_STACK_KERNELS(bfloat16);

}  // namespace musa
}  // namespace tensorflow