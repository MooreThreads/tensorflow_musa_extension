#include <musa_runtime_api.h>

#include <limits>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchMusaTileKernel(const T* input, const int64_t* input_dims,
                          const int64_t* output_dims, int dims,
                          int64_t output_size, T* output, musaStream_t stream);

namespace {

template <typename T, typename Tmultiples>
class MusaTileOp : public MusaOpKernel {
 public:
  explicit MusaTileOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(multiples.shape()),
                errors::InvalidArgument("multiples must be a 1-D tensor"));

    const int input_dims = input.dims();
    OP_REQUIRES(context, multiples.NumElements() == input_dims,
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input_dims, " but got length ", multiples.NumElements()));

    const Tmultiples* m_data = multiples.flat<Tmultiples>().data();

    TensorShape output_shape;
    bool need_tile = false;
    std::vector<int64_t> input_dims_host(input_dims);
    std::vector<int64_t> output_dims_host(input_dims);

    for (int i = 0; i < input_dims; ++i) {
      const int64_t in_dim = input.dim_size(i);
      const int64_t multiple = static_cast<int64_t>(m_data[i]);
      OP_REQUIRES(context, multiple >= 0,
                  errors::InvalidArgument("Expected multiples[", i,
                                          "] >= 0, got ", multiple));
      OP_REQUIRES(
          context,
          in_dim == 0 ||
              multiple <= std::numeric_limits<int64_t>::max() / in_dim,
          errors::InvalidArgument("Tile output dimension overflow at dim ", i));

      const int64_t out_dim = in_dim * multiple;
      output_shape.AddDim(out_dim);
      input_dims_host[i] = in_dim;
      output_dims_host[i] = out_dim;
      if (multiple != 1) need_tile = true;
    }

    if (input_dims == 0 || !need_tile) {
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    Tensor input_dims_dev;
    Tensor output_dims_dev;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({input_dims}),
                                          &input_dims_dev));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({input_dims}),
                                          &output_dims_dev));

    musaStream_t stream = GetMusaStreamByCtx(context);
    const size_t dims_bytes = input_dims * sizeof(int64_t);

    auto copy_status = musaMemcpyAsync(input_dims_dev.flat<int64_t>().data(),
                                       input_dims_host.data(), dims_bytes,
                                       musaMemcpyHostToDevice, stream);
    OP_REQUIRES(context, copy_status == musaSuccess,
                errors::Internal("Tile input_dims H2D copy failed: ",
                                 static_cast<int>(copy_status)));

    copy_status = musaMemcpyAsync(output_dims_dev.flat<int64_t>().data(),
                                  output_dims_host.data(), dims_bytes,
                                  musaMemcpyHostToDevice, stream);
    OP_REQUIRES(context, copy_status == musaSuccess,
                errors::Internal("Tile output_dims H2D copy failed: ",
                                 static_cast<int>(copy_status)));

    LaunchMusaTileKernel<T>(
        input.flat<T>().data(), input_dims_dev.flat<int64_t>().data(),
        output_dims_dev.flat<int64_t>().data(), input_dims,
        output->NumElements(), output->flat<T>().data(), stream);
  }
};

#define REGISTER_MUSA_TILE_ALL_TYPES(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          MusaTileOp<type, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          MusaTileOp<type, int64>);

REGISTER_MUSA_TILE_ALL_TYPES(float);
REGISTER_MUSA_TILE_ALL_TYPES(Eigen::half);
REGISTER_MUSA_TILE_ALL_TYPES(double);
REGISTER_MUSA_TILE_ALL_TYPES(int32);
REGISTER_MUSA_TILE_ALL_TYPES(int64);
REGISTER_MUSA_TILE_ALL_TYPES(bool);

#undef REGISTER_MUSA_TILE_ALL_TYPES

}  // namespace
}  // namespace musa
}  // namespace tensorflow
