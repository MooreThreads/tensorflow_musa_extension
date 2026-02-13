#include <musa_runtime_api.h>

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
class MusaArgMaxOp : public OpKernel {
 public:
  explicit MusaArgMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axis_tensor = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be scalar"));

    int64_t axis = 0;
    if (axis_tensor.dtype() == DT_INT32) {
      axis = axis_tensor.scalar<int32>()();
    } else {
      axis = axis_tensor.scalar<int64>()();
    }

    const int input_dims = input.dims();
    OP_REQUIRES(ctx, axis >= -input_dims && axis < input_dims,
                errors::InvalidArgument("Expected axis in range [", -input_dims,
                                        ", ", input_dims, "), but got ", axis));
    if (axis < 0) axis += input_dims;

    const int64_t axis_size = input.dim_size(axis);
    OP_REQUIRES(ctx, axis_size > 0,
                errors::InvalidArgument(
                    "Reduction axis ", axis,
                    " is empty in shape: ", input.shape().DebugString()));

    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      if (i != axis) output_shape.AddDim(input.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() == 0) return;

    size_t input_bytes = input.NumElements() * sizeof(T);
    std::vector<T> h_input(input.NumElements());
    auto status = musaMemcpy(h_input.data(), input.flat<T>().data(),
                             input_bytes, musaMemcpyDeviceToHost);
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaArgMax: Memcpy D2H failed"));

    std::vector<Tidx> h_output(output->NumElements());

    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= input.dim_size(i);
    int64_t inner_size = 1;
    for (int i = axis + 1; i < input_dims; ++i) inner_size *= input.dim_size(i);

    for (int64_t o = 0; o < outer_size; ++o) {
      for (int64_t i = 0; i < inner_size; ++i) {
        float max_val = -1e30;
        Tidx max_idx = 0;

        for (int64_t a = 0; a < axis_size; ++a) {
          int64_t input_idx = o * (axis_size * inner_size) + a * inner_size + i;

          float val = static_cast<float>(h_input[input_idx]);

          if (a == 0 || val > max_val) {
            max_val = val;
            max_idx = static_cast<Tidx>(a);
          }
        }

        h_output[o * inner_size + i] = max_idx;
      }
    }

    size_t output_bytes = output->NumElements() * sizeof(Tidx);
    status = musaMemcpy(output->flat<Tidx>().data(), h_output.data(),
                        output_bytes, musaMemcpyHostToDevice);
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaArgMax: Memcpy H2D failed"));
  }
};

#define REGISTER_MUSA_ARGMAX(T, Tidx)                              \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                           \
                              .Device("MUSA")                      \
                              .TypeConstraint<T>("T")              \
                              .TypeConstraint<Tidx>("output_type") \
                              .HostMemory("dimension"),            \
                          MusaArgMaxOp<T, Tidx>);

REGISTER_MUSA_ARGMAX(float, int64);
REGISTER_MUSA_ARGMAX(float, int32);

REGISTER_MUSA_ARGMAX(int32, int64);
REGISTER_MUSA_ARGMAX(int32, int32);

REGISTER_MUSA_ARGMAX(Eigen::half, int64);
REGISTER_MUSA_ARGMAX(Eigen::half, int32);

REGISTER_MUSA_ARGMAX(bfloat16, int64);
REGISTER_MUSA_ARGMAX(bfloat16, int32);

REGISTER_MUSA_ARGMAX(double, int64);
REGISTER_MUSA_ARGMAX(double, int32);

#undef REGISTER_MUSA_ARGMAX

}  // namespace musa
}  // namespace tensorflow
