#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace musa {

class MusaConstOp : public OpKernel {
 public:
  explicit MusaConstOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), initialized_(false) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    OP_REQUIRES(ctx, cpu_tensor_.FromProto(*proto),
                errors::InvalidArgument("Unparseable tensor proto"));
    OP_REQUIRES(
        ctx, cpu_tensor_.dtype() == ctx->output_type(0),
        errors::InvalidArgument("Type mismatch between value and output"));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    if (cpu_tensor_.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, cpu_tensor_.shape(), &output));
      return;
    }

    // Lazy Initialization
    {
      mutex_lock lock(mu_);
      if (!initialized_) {
        AllocatorAttributes attr;
        attr.set_on_host(false);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(cpu_tensor_.dtype(), cpu_tensor_.shape(),
                                    &gpu_tensor_, attr));

        auto& handle = GetHandleByCtx(ctx);
        musaStream_t stream =
            reinterpret_cast<musaStream_t>(handle.GetStream());
        size_t total_bytes = cpu_tensor_.TotalBytes();

        // only perform H2D copy once during the first execution
        musaError_t err =
            musaMemcpyAsync(const_cast<char*>(gpu_tensor_.tensor_data().data()),
                            cpu_tensor_.tensor_data().data(), total_bytes,
                            musaMemcpyHostToDevice, stream);

        OP_REQUIRES(ctx, err == musaSuccess,
                    errors::Internal("MUSA Const H2D Memcpy failed: ",
                                     musaGetErrorString(err)));
        initialized_ = true;
      }
    }

    ctx->set_output(0, gpu_tensor_);
  }

 private:
  Tensor cpu_tensor_;
  Tensor gpu_tensor_;
  bool initialized_;
  mutex mu_;
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
