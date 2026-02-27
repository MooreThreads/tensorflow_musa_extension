#include <mudnn.h>

#include <functional>
#include <memory>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaMean")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("Tidx: {int32, int64}")
    .SetShapeFn(shape_inference::ReductionShape);

template <typename T, typename Tidx>
class MusaMeanOp : public MusaOpKernel {
 public:
  explicit MusaMeanOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
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

    if (num_axes == 0) {
      for (int i = 0; i < input.dims(); ++i) {
        bitmap[i] = true;
        reduce_dims.push_back(i);
      }
    } else {
      auto axes_flat = axes_tensor.flat<Tidx>();
      for (int64_t i = 0; i < num_axes; i++) {
        Tidx index = axes_flat(i);
        if (index < 0) index += input.dims();
        if (index >= 0 && index < input.dims() && !bitmap[index]) {
          bitmap[index] = true;
          reduce_dims.push_back(static_cast<int>(index));
        }
      }
    }

    TensorShape output_shape;
    TensorShape musa_output_shape;
    int64_t reduce_elements = 1;

    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        reduce_elements *= input.dim_size(d);
        if (keep_dims_) output_shape.AddDim(1);
        musa_output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input.dim_size(d));
        musa_output_shape.AddDim(input.dim_size(d));
      }
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (out->NumElements() == 0 || reduce_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    if (reduce_elements == 1) {
      musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
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
    op.SetMode(::musa::dnn::Reduce::Mode::MEAN);
    op.SetDim(reduce_dims.size(), reduce_dims.data());

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());

    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      return std::unique_ptr<void, std::function<void(void*)>>(
          ptr, [tf_allocator](void* p) {
            if (p) tf_allocator->DeallocateRaw(p);
          });
    };

    ::musa::dnn::MemoryMaintainer mm(alloc_func);
    auto status = op.Run(handle, t_out, t_in, mm);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Mean execution failed. Status: ", (int)status));
  }

 private:
  bool keep_dims_;
};

#define REGISTER_MEAN_KERNEL(T, Tidx)                           \
  REGISTER_KERNEL_BUILDER(Name("Mean")                          \
                              .Device("MUSA")                   \
                              .TypeConstraint<T>("T")           \
                              .TypeConstraint<Tidx>("Tidx")     \
                              .HostMemory("reduction_indices"), \
                          MusaMeanOp<T, Tidx>);                 \
  REGISTER_KERNEL_BUILDER(Name("MusaMean")                      \
                              .Device("MUSA")                   \
                              .TypeConstraint<T>("T")           \
                              .TypeConstraint<Tidx>("Tidx")     \
                              .HostMemory("reduction_indices"), \
                          MusaMeanOp<T, Tidx>);

REGISTER_MEAN_KERNEL(float, int32);
REGISTER_MEAN_KERNEL(float, int64);
REGISTER_MEAN_KERNEL(Eigen::half, int32);
REGISTER_MEAN_KERNEL(Eigen::half, int64);
REGISTER_MEAN_KERNEL(bfloat16, int32);
REGISTER_MEAN_KERNEL(bfloat16, int64);

#undef REGISTER_MEAN_KERNEL

}  // namespace musa
}  // namespace tensorflow
