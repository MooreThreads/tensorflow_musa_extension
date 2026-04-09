#include <mudnn.h>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "musa_reduce_functor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

template <typename Tidx>
class MusaAllOp : public MusaOpKernel {
 public:
  explicit MusaAllOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* context) override {
    MUSA_DEBUG_LOG_KERNEL(context);
    const Tensor& input = context->input(0);
    const Tensor& axes_tensor = context->input(1);

    if (input.NumElements() == 0) {
      context->set_output(0, input);
      return;
    }

    int64_t num_axes = axes_tensor.NumElements();
    std::vector<int> reduce_dims;
    std::vector<bool> bitmap(input.dims(), false);

    // reduce dim to construct `reduce_dims`, which would be feed into
    // ReduceFunctor (required by mReduce)
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
      }
    } else {
      for (int i = 0; i < input.dims(); ++i) {
        bitmap[i] = true;
        reduce_dims.push_back(i);
      }
    }

    TensorShape output_shape;
    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        if (keep_dims_) output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input.dim_size(d));
      }
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    OP_REQUIRES_OK(
        context,
        ReduceFunctor::Compute<Tidx>(
            context, &mt_output, &mt_input, ::musa::dnn::Reduce::Mode::AND,
            reduce_dims.data(), reduce_dims.size(), "MusaAllOp Run failed: "));
  }

 private:
  bool keep_dims_;
};

#define REGISTER_MUSA_ALL_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("All")                          \
                              .Device("MUSA")                  \
                              .HostMemory("reduction_indices") \
                              .TypeConstraint<type>("Tidx"),   \
                          MusaAllOp<type>)

REGISTER_MUSA_ALL_KERNEL(bool);
REGISTER_MUSA_ALL_KERNEL(int32);
REGISTER_MUSA_ALL_KERNEL(int64);
REGISTER_MUSA_ALL_KERNEL(float);
REGISTER_MUSA_ALL_KERNEL(double);
REGISTER_MUSA_ALL_KERNEL(bfloat16);
REGISTER_MUSA_ALL_KERNEL(Eigen::half);

}  // namespace musa
}  // namespace tensorflow
