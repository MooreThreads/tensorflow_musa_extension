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
    OP_REQUIRES(
        context,
        input.NumElements() <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        errors::InvalidArgument("WhereOp: input is too large, num_elements=",
                                input.NumElements()));
    ComputeType<int32>(context, input, input_dims);
  }

  template <typename Tindex>
  void ComputeType(OpKernelContext* context, const Tensor& input,
                   int input_dims) {
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);

    Tensor num_true_tensor;
    OP_REQUIRES_OK(context,
             context->allocate_temp(DataTypeToEnum<Tindex>::value,
                        TensorShape({1}), &num_true_tensor,
                        alloc_attr));
    typename TTypes<Tindex>::UnalignedScalar num_true_t(
      num_true_tensor.flat<Tindex>().data());

    Status s =
        NumTrue<T, Tindex>::Compute(context, input.flat<T>(), num_true_t);
    OP_REQUIRES_OK(context, s);

    auto create_and_check_output = [context, &input, input_dims,
                    num_true_tensor =
                      std::move(num_true_tensor)]() {
      const Tindex num_true = *num_true_tensor.flat<Tindex>().data();
      Tensor* output = nullptr;

      TensorShape output_shape;
      OP_REQUIRES_OK(context,
             output_shape.AddDimWithStatus(static_cast<int64>(num_true)));
      OP_REQUIRES_OK(
        context,
        output_shape.AddDimWithStatus(static_cast<int64>(input_dims)));
      OP_REQUIRES_OK(
          context,
        context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                              \
  case NDIM: {                                                        \
    Status where_status = Where::Compute<NDIM, T, Tindex>(            \
        context, input.tensor<T, NDIM>(), output->matrix<int64_t>()); \
    OP_REQUIRES_OK(context, where_status);                            \
                                                                      \
  } break
      switch (input_dims) {
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
    };

    musaStream_t stream = GetMusaStreamByCtx(context);
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, create_and_check_output);
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
