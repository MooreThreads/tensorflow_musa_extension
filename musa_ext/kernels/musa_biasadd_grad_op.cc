#include <mudnn.h>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaBiasAddGradOp : public MusaOpKernel {
 public:
  explicit MusaBiasAddGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& output_backprop = ctx->input(0);

    OP_REQUIRES(ctx, output_backprop.dims() >= 1,
                errors::InvalidArgument("Input tensor must be at least 1D: ",
                                        output_backprop.shape().DebugString()));

    int channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;
    } else {
      channel_dim = output_backprop.dims() - 1;
    }

    if (channel_dim < 0) channel_dim += output_backprop.dims();
    OP_REQUIRES(
        ctx, channel_dim >= 0 && channel_dim < output_backprop.dims(),
        errors::InvalidArgument("Invalid channel dimension calculation."));

    TensorShape output_shape({output_backprop.dim_size(channel_dim)});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_backprop.NumElements() == 0) return;

    std::vector<int> reduce_dims;
    reduce_dims.reserve(output_backprop.dims() - 1);
    for (int i = 0; i < output_backprop.dims(); ++i) {
      if (i != channel_dim) {
        reduce_dims.push_back(i);
      }
    }

    auto& handle = GetHandleByCtx(ctx);

    if (reduce_dims.empty()) {
      musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
      MusaMemcpyAsyncD2D(const_cast<char*>(output->tensor_data().data()),
                         output_backprop.tensor_data().data(),
                         output_backprop.TotalBytes(), stream);
      return;
    }

    TensorShape mudnn_output_shape = output_backprop.shape();
    for (int dim : reduce_dims) {
      mudnn_output_shape.set_dim(dim, 1);
    }

    Tensor output_reshaped;
    OP_REQUIRES(ctx, output_reshaped.CopyFrom(*output, mudnn_output_shape),
                errors::Internal("Failed to reshape output for muDNN"));

    mTensor t_in = CreateMTensor(output_backprop, format_);
    mTensor t_out = CreateMTensor(output_reshaped, format_);

    mReduce op;
    op.SetMode(::musa::dnn::Reduce::Mode::ADD);
    op.SetDim(reduce_dims.size(), reduce_dims.data());

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      std::function<void(void*)> deleter = [tf_allocator](void* p) {
        if (p) tf_allocator->DeallocateRaw(p);
      };
      return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };
    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status = op.Run(handle, t_out, t_in, mm);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA BiasAddGrad failed. Status: ", (int)status));
  }

 private:
  TensorFormat data_format_;
};

#define REGISTER_MUSA_BIAS_ADD_GRAD(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("BiasAddGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaBiasAddGradOp<TYPE>);

REGISTER_MUSA_BIAS_ADD_GRAD(float);
REGISTER_MUSA_BIAS_ADD_GRAD(double);
REGISTER_MUSA_BIAS_ADD_GRAD(Eigen::half);
REGISTER_MUSA_BIAS_ADD_GRAD(bfloat16);
REGISTER_MUSA_BIAS_ADD_GRAD(int32);
REGISTER_MUSA_BIAS_ADD_GRAD(int64);

}  // namespace musa
}  // namespace tensorflow
