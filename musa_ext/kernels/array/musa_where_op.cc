#include "musa_where_op.h"

#include <limits>
#include <utility>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaWhereOp : public MusaOpKernel {
 public:
  explicit MusaWhereOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();
    if (input.NumElements() == 0) {
      // Handle the case where there are no elements in the input tensor.
      Tensor* out = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({0, input_dims}), &out));
      return;
    }

    if (input.NumElements() < std::numeric_limits<int32_t>::max()) {
      ComputeType<int32_t>(context, input, input_dims);
    } else {
      ComputeType<int64_t>(context, input, input_dims);
    }
  }

  template <typename Tindex>
  void ComputeType(OpKernelContext* context, const Tensor& input,
                   int input_dims) {
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);

    // We first need to count the number of true elements in the input tensor,
    // which will determine the output shape.
    Tensor num_true_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<int64>::value, TensorShape({1}),
                                &num_true_tensor, alloc_attr));
    typename TTypes<int64>::UnalignedScalar num_true_t(
        num_true_tensor.flat<int64>().data());

    Status s = NumTrue<T, int64>::Compute(context, input.flat<T>(), num_true_t);
    OP_REQUIRES_OK(context, s);

    const int64 num_true = *num_true_tensor.flat<int64>().data();
    if (num_true == 0) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({0, input_dims}), &out));
      return;
    }

    // Next is to compute `where`, given the number of true elements.
    Tensor* output = nullptr;
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   output_shape.AddDimWithStatus(static_cast<int64>(num_true)));
    OP_REQUIRES_OK(
        context, output_shape.AddDimWithStatus(static_cast<int64>(input_dims)));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                            \
  case NDIM: {                                                      \
    Status where_status = Where::Compute<NDIM, T, int64>(           \
        context, input.tensor<T, NDIM>(), output->matrix<int64>()); \
    OP_REQUIRES_OK(context, where_status);                          \
                                                                    \
  } break
    switch (input_dims) {
      case 0:
        break; // For a scalar input, output shape is [num_true, 0]. No coordinates to write.
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp: Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM
  }

  bool IsExpensive() override { return true; }
};

#define REGISTER_MUSA_WHERE_OP(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Where").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaWhereOp<TYPE>)

REGISTER_MUSA_WHERE_OP(float);
REGISTER_MUSA_WHERE_OP(double);
REGISTER_MUSA_WHERE_OP(int8);
REGISTER_MUSA_WHERE_OP(uint8);
REGISTER_MUSA_WHERE_OP(int16);
REGISTER_MUSA_WHERE_OP(uint16);
REGISTER_MUSA_WHERE_OP(int32);
REGISTER_MUSA_WHERE_OP(int64);
REGISTER_MUSA_WHERE_OP(bfloat16);
REGISTER_MUSA_WHERE_OP(bool);

#undef REGISTER_MUSA_WHERE_OP

}  // namespace musa
}  // namespace tensorflow
