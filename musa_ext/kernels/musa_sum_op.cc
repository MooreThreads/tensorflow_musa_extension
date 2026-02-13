#include <mudnn.h>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSumOp : public MusaOpKernel {
 public:
  explicit MusaSumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axes_tensor = ctx->input(1);

    if (input.NumElements() == 0) {
      ctx->set_output(0, input);
      return;
    }

    int64_t num_axes = axes_tensor.NumElements();
    std::vector<int> reduce_dims;
    gtl::InlinedVector<bool, 4> bitmap(input.dims(), false);

    if (num_axes > 0) {
      if (axes_tensor.dtype() == DT_INT32) {
        auto axes_flat = axes_tensor.flat<int32>();
        for (int64_t i = 0; i < num_axes; ++i) {
          int32 index = axes_flat(i);
          if (index < 0) index += input.dims();
          if (index >= 0 && index < input.dims() && !bitmap[index]) {
            bitmap[index] = true;
            reduce_dims.push_back(static_cast<int>(index));
          }
        }
      } else if (axes_tensor.dtype() == DT_INT64) {
        auto axes_flat = axes_tensor.flat<int64>();
        for (int64_t i = 0; i < num_axes; ++i) {
          int64 index = axes_flat(i);
          if (index < 0) index += input.dims();
          if (index >= 0 && index < input.dims() && !bitmap[index]) {
            bitmap[index] = true;
            reduce_dims.push_back(static_cast<int>(index));
          }
        }
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument(
                        "reduction_indices must be int32 or int64"));
      }
    }

    TensorShape output_shape;
    TensorShape musa_output_shape;
    int64_t reduce_elements = 1;

    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        reduce_elements *= input.dim_size(d);
        if (keep_dims_) {
          output_shape.AddDim(1);
        }
        musa_output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input.dim_size(d));
        musa_output_shape.AddDim(input.dim_size(d));
      }
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (out->NumElements() == 0) return;

    if (reduce_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    if (reduce_elements == 1) {
      MusaMemcpyAsyncD2D(const_cast<char*>(out->tensor_data().data()),
                         input.tensor_data().data(), input.TotalBytes(),
                         stream);
      return;
    }

    Tensor out_reshaped(out->dtype());
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(*out, musa_output_shape),
                errors::Internal("Reshape failed."));

    mTensor t_in = CreateMTensor(input, format_);
    mTensor t_out = CreateMTensor(out_reshaped, format_);

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
        errors::Internal("MUSA muDNN Reduce Sum execution failed. Status: ",
                         (int)status));
  }

 private:
  bool keep_dims_;
};

#define REGISTER_MUSA_SUM(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("Sum")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int32>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaSumOp<TYPE>);                     \
  REGISTER_KERNEL_BUILDER(Name("Sum")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int64>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaSumOp<TYPE>);

REGISTER_MUSA_SUM(float);
REGISTER_MUSA_SUM(Eigen::half);
REGISTER_MUSA_SUM(bfloat16);
REGISTER_MUSA_SUM(double);
REGISTER_MUSA_SUM(int32);
REGISTER_MUSA_SUM(int64);

#undef REGISTER_MUSA_SUM

}  // namespace musa
}  // namespace tensorflow
