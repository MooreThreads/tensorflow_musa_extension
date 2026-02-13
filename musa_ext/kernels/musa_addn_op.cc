#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaAddNOp : public MusaOpKernel {
 public:
  explicit MusaAddNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const int num_inputs = ctx->num_inputs();
    OP_REQUIRES(ctx, num_inputs >= 1,
                errors::InvalidArgument("AddN requires at least one input."));

    if (num_inputs == 1) {
      const Tensor& input = ctx->input(0);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

      if (input.NumElements() == 0) {
        return;
      }

      auto& handle = GetHandleByCtx(ctx);

      mTensor t_input = CreateMTensor(input, format_);
      mTensor t_output = CreateMTensor(*output, format_);

      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::ADD);

      auto status = op.Run(handle, t_output, t_input, t_input);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MUSA AddN single input execution failed."));
      return;
    }

    const Tensor& input0 = ctx->input(0);
    TensorShape output_shape = input0.shape();

    for (int i = 1; i < num_inputs; ++i) {
      const Tensor& input_i = ctx->input(i);
      OP_REQUIRES(
          ctx, input_i.shape() == output_shape,
          errors::InvalidArgument(
              "All inputs to AddN must have the same shape. Input 0 shape: ",
              output_shape.DebugString(), " vs input ", i,
              " shape: ", input_i.shape().DebugString()));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);

    if (num_inputs == 2) {
      const Tensor& input1 = ctx->input(1);

      mTensor t0 = CreateMTensor(input0, format_);
      mTensor t1 = CreateMTensor(input1, format_);
      mTensor t_out = CreateMTensor(*output, format_);

      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::ADD);

      auto status = op.Run(handle, t_out, t0, t1);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MUSA AddN two inputs execution failed."));
      return;
    }

    Tensor temp_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input0.dtype(), output_shape, &temp_tensor));

    mTensor t0 = CreateMTensor(input0, format_);
    mTensor t1 = CreateMTensor(ctx->input(1), format_);
    mTensor t_temp = CreateMTensor(temp_tensor, format_);

    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::ADD);

    auto status = op.Run(handle, t_temp, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA AddN intermediate addition failed."));

    for (int i = 2; i < num_inputs - 1; ++i) {
      mTensor t_input = CreateMTensor(ctx->input(i), format_);

      Tensor next_temp;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(input0.dtype(), output_shape, &next_temp));
      mTensor t_next_temp = CreateMTensor(next_temp, format_);

      status = op.Run(handle, t_next_temp, t_temp, t_input);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("MUSA AddN intermediate addition failed."));

      temp_tensor = next_temp;
      t_temp = t_next_temp;
    }

    mTensor t_last_input = CreateMTensor(ctx->input(num_inputs - 1), format_);
    mTensor t_out = CreateMTensor(*output, format_);

    status = op.Run(handle, t_out, t_temp, t_last_input);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA AddN final addition failed."));
  }
};

#define REGISTER_MUSA_ADDN(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("AddN").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaAddNOp<TYPE>);

REGISTER_MUSA_ADDN(float);
REGISTER_MUSA_ADDN(double);
REGISTER_MUSA_ADDN(Eigen::half);
REGISTER_MUSA_ADDN(bfloat16);
REGISTER_MUSA_ADDN(int32);
REGISTER_MUSA_ADDN(int64);

}  // namespace musa
}  // namespace tensorflow
