#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaFloorModOp : public MusaOpKernel {
 public:
  explicit MusaFloorModOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " and ",
                    in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (in0.NumElements() == 0 || in1.NumElements() == 0 ||
        output_shape.num_elements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);

    mTensor t0 = CreateMTensor(in0, format_);
    mTensor t1 = CreateMTensor(in1, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    if (!bcast.IsBroadcastingRequired()) {
      compute_floormod(handle, t_out, t0, t1, output_shape, in0.dtype(), ctx);
    } else {
      compute_broadcast_floormod(handle, t_out, t0, t1, in0, in1, bcast,
                                 output_shape, in0.dtype(), ctx);
    }
  }

 private:
  void compute_floormod(::musa::dnn::Handle& handle, mTensor& t_out,
                        mTensor& t0, mTensor& t1,
                        const TensorShape& output_shape, DataType dtype,
                        OpKernelContext* ctx) {
    Tensor temp_div;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_div));
    mTensor t_temp_div = CreateMTensor(temp_div, format_);

    ::musa::dnn::Binary div_op;
    div_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    auto status = div_op.Run(handle, t_temp_div, t0, t1);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Division execution failed in FloorMod."));

    Tensor temp_floor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_floor));
    mTensor t_temp_floor = CreateMTensor(temp_floor, format_);

    ::musa::dnn::Unary floor_op;
    floor_op.SetMode(::musa::dnn::Unary::Mode::FLOOR);
    status = floor_op.Run(handle, t_temp_floor, t_temp_div);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Floor execution failed in FloorMod."));

    Tensor temp_mul;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_mul));
    mTensor t_temp_mul = CreateMTensor(temp_mul, format_);

    ::musa::dnn::Binary mul_op;
    mul_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    status = mul_op.Run(handle, t_temp_mul, t_temp_floor, t1);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Multiplication execution failed in FloorMod."));

    ::musa::dnn::Binary sub_op;
    sub_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    status = sub_op.Run(handle, t_out, t0, t_temp_mul);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Subtraction execution failed in FloorMod."));
  }

  void compute_broadcast_floormod(::musa::dnn::Handle& handle, mTensor& t_out,
                                  mTensor& t0, mTensor& t1, const Tensor& in0,
                                  const Tensor& in1, const BCast& bcast,
                                  const TensorShape& output_shape,
                                  DataType dtype, OpKernelContext* ctx) {
    Tensor temp_div;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_div));
    mTensor t_temp_div = CreateMTensor(temp_div, format_);

    ::musa::dnn::Binary div_op;
    div_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    auto status = div_op.Run(handle, t_temp_div, t0, t1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA Division execution failed in FloorMod (broadcast)."));

    Tensor temp_floor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_floor));
    mTensor t_temp_floor = CreateMTensor(temp_floor, format_);

    ::musa::dnn::Unary floor_op;
    floor_op.SetMode(::musa::dnn::Unary::Mode::FLOOR);
    status = floor_op.Run(handle, t_temp_floor, t_temp_div);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal(
                    "MUSA Floor execution failed in FloorMod (broadcast)."));

    Tensor temp_mul;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, output_shape, &temp_mul));
    mTensor t_temp_mul = CreateMTensor(temp_mul, format_);

    ::musa::dnn::Binary mul_op;
    mul_op.SetMode(::musa::dnn::Binary::Mode::MUL);
    status = mul_op.Run(handle, t_temp_mul, t_temp_floor, t1);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal(
            "MUSA Multiplication execution failed in FloorMod (broadcast)."));

    ::musa::dnn::Binary sub_op;
    sub_op.SetMode(::musa::dnn::Binary::Mode::SUB);
    status = sub_op.Run(handle, t_out, t0, t_temp_mul);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal(
            "MUSA Subtraction execution failed in FloorMod (broadcast)."));
  }
};

#define REGISTER_MUSA_FLOORMOD(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FloorMod").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFloorModOp<TYPE>);

REGISTER_MUSA_FLOORMOD(float);
REGISTER_MUSA_FLOORMOD(double);
REGISTER_MUSA_FLOORMOD(Eigen::half);
REGISTER_MUSA_FLOORMOD(bfloat16);
REGISTER_MUSA_FLOORMOD(int32);
REGISTER_MUSA_FLOORMOD(int64);

}  // namespace musa
}  // namespace tensorflow
