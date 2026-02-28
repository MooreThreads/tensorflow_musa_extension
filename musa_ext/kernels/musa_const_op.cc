#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/version.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaConstOp : public OpKernel {
 public:
  explicit MusaConstOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    OP_REQUIRES(ctx, cpu_tensor_.FromProto(*proto),
                errors::InvalidArgument("Unparseable tensor proto"));
    OP_REQUIRES(
        ctx, cpu_tensor_.dtype() == ctx->output_type(0),
        errors::InvalidArgument("Type mismatch between value and output"));

    // Delayed initialization flag
    gpu_buffer_initialized_ = false;
    gpu_buffer_ = nullptr;
  }

  // Const op is inexpensive (simple D2D memcpy after first initialization)
  // Marking as inexpensive enables TensorFlow executor inline scheduling
  bool IsExpensive() override { return false; }

  ~MusaConstOp() override {
    if (gpu_buffer_ != nullptr) {
      musaFree(gpu_buffer_);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, cpu_tensor_.shape(), &output));
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    void* dst_ptr = const_cast<char*>(output->tensor_data().data());
    size_t total_bytes = cpu_tensor_.TotalBytes();

    // Delayed initialization: allocate GPU memory and copy on the first call
    if (!gpu_buffer_initialized_) {
      musaError_t err = musaMalloc(&gpu_buffer_, total_bytes);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("MUSA Const malloc failed: ",
                                   musaGetErrorString(err)));

      err = musaMemcpy(gpu_buffer_, cpu_tensor_.tensor_data().data(),
                       total_bytes, musaMemcpyHostToDevice);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("MUSA Const H2D Memcpy failed: ",
                                   musaGetErrorString(err)));

      gpu_buffer_initialized_ = true;
    }

    // 使用 D2D 拷贝，性能优于 H2D
    musaError_t err =
        musaMemcpyAsync(dst_ptr, gpu_buffer_, total_bytes, musaMemcpyDeviceToDevice,
                        (musaStream_t)handle.GetStream());
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA Const D2D Memcpy failed: ",
                                 musaGetErrorString(err)));
  }

 private:
  Tensor cpu_tensor_;
  void* gpu_buffer_;
  bool gpu_buffer_initialized_;
};

#define REGISTER_MUSA_CONST(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Const").Device(DEVICE_MTGPU).TypeConstraint<type>("dtype"), \
      MusaConstOp);

REGISTER_MUSA_CONST(float);
REGISTER_MUSA_CONST(double);
REGISTER_MUSA_CONST(Eigen::half);
REGISTER_MUSA_CONST(bfloat16);
REGISTER_MUSA_CONST(int64);
REGISTER_MUSA_CONST(int32);
REGISTER_MUSA_CONST(int16);
REGISTER_MUSA_CONST(int8);
REGISTER_MUSA_CONST(uint64);
REGISTER_MUSA_CONST(uint32);
REGISTER_MUSA_CONST(uint16);
REGISTER_MUSA_CONST(uint8);
REGISTER_MUSA_CONST(bool);
REGISTER_MUSA_CONST(std::complex<float>);
REGISTER_MUSA_CONST(std::complex<double>);

#undef REGISTER_MUSA_CONST

}  // namespace musa
}  // namespace tensorflow
