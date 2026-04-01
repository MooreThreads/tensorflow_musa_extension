#include <mudnn.h>

#include <functional>
#include <numeric>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMaxOp : public MusaOpKernel {
 public:
  explicit MusaMaxOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axes_tensor = ctx->input(1);

    const int rank = input.dims();
    const int64_t num_axes = axes_tensor.NumElements();

    if (num_axes == 0) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &out));
      if (out->NumElements() == 0) return;

      auto& handle = GetHandleByCtx(ctx);
      musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
      MusaMemcpyAsyncD2D(const_cast<char*>(out->tensor_data().data()),
                         input.tensor_data().data(), out->TotalBytes(), stream);
      return;
    }

    std::vector<int> reduce_dims;
    gtl::InlinedVector<bool, 4> bitmap(rank, false);

    auto push_axis = [&](int64_t a64) {
      if (a64 < 0) a64 += rank;
      OP_REQUIRES(ctx, a64 >= 0 && a64 < rank,
                  errors::InvalidArgument("axis out of range: ", a64,
                                          " for rank ", rank));
      int a = static_cast<int>(a64);
      if (!bitmap[a]) {
        bitmap[a] = true;
        reduce_dims.push_back(a);
      }
    };

    if (axes_tensor.dtype() == DT_INT32) {
      auto axes_flat = axes_tensor.flat<int32>();
      for (int64_t i = 0; i < num_axes; ++i) push_axis(axes_flat(i));
    } else if (axes_tensor.dtype() == DT_INT64) {
      auto axes_flat = axes_tensor.flat<int64>();
      for (int64_t i = 0; i < num_axes; ++i) push_axis(axes_flat(i));
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("reduction_indices must be int32 or int64"));
    }

    TensorShape output_shape;
    TensorShape musa_output_shape;
    int64_t reduce_elements = 1;

    for (int d = 0; d < rank; ++d) {
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
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    if (reduce_elements == 1) {
      MusaMemcpyAsyncD2D(const_cast<char*>(out->tensor_data().data()),
                         input.tensor_data().data(), out->TotalBytes(), stream);
      return;
    }

    Tensor out_reshaped(out->dtype());
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(*out, musa_output_shape),
                errors::Internal("Reshape failed."));

    mTensor t_in = CreateMTensor(input, format_);
    mTensor t_out = CreateMTensor(out_reshaped, format_);

    ::musa::dnn::Reduce op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Reduce::Mode::MAX), "Set Reduce MAX",
                  ctx);
    MTOP_CHECK_OK(
        op.SetDim(static_cast<int>(reduce_dims.size()), reduce_dims.data()),
        "Set Reduce Dims", ctx);

    // MemoryMaintainer: unique_ptr<void, function<...>> + std::function factory
    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());

    auto alloc_func =
        [tf_allocator](size_t size) -> ::musa::dnn::MemoryHandler {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      std::function<void(void*)> deleter = [tf_allocator](void* p) {
        if (p) tf_allocator->DeallocateRaw(p);
      };
      return ::musa::dnn::MemoryHandler(ptr, deleter);
    };

    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status = op.Run(handle, t_out, t_in, mm);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA muDNN Reduce Max execution failed. Status: ",
                         static_cast<int>(status)));
  }

 private:
  bool keep_dims_ = false;
};

#define REGISTER_MUSA_MAX(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("Max")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int32>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaMaxOp<TYPE>);                     \
  REGISTER_KERNEL_BUILDER(Name("Max")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int64>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaMaxOp<TYPE>);

REGISTER_MUSA_MAX(float);
REGISTER_MUSA_MAX(Eigen::half);
REGISTER_MUSA_MAX(bfloat16);
REGISTER_MUSA_MAX(double);
REGISTER_MUSA_MAX(int32);
REGISTER_MUSA_MAX(int64);

#undef REGISTER_MUSA_MAX

}  // namespace musa
}  // namespace tensorflow