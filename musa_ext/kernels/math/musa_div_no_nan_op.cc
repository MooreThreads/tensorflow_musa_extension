#include <musa_runtime.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

struct DivNoNanStrides {
  int s0, s1, s2, s3;
};

struct DivNoNanDims {
  int d0, d1, d2, d3;
};

template <typename T>
void LaunchDivNoNan(const T* in0, const T* in1, T* out,
                    DivNoNanStrides s_in0, DivNoNanStrides s_in1,
                    DivNoNanDims dims, int total_elements,
                    musaStream_t stream);

namespace {

TensorShape PadShapeTo4D(const TensorShape& shape) {
  TensorShape padded = shape;
  while (padded.dims() < 4) padded.InsertDim(0, 1);
  return padded;
}

DivNoNanStrides CalcDivNoNanStrides(const TensorShape& shape,
                                    const TensorShape& output_shape) {
  int64_t raw_strides[4];
  int64_t acc = 1;
  for (int i = 3; i >= 0; --i) {
    raw_strides[i] = acc;
    acc *= shape.dim_size(i);
  }

  DivNoNanStrides strides;
  int* stride_values = reinterpret_cast<int*>(&strides);
  for (int i = 0; i < 4; ++i) {
    stride_values[i] =
        (shape.dim_size(i) == 1 && output_shape.dim_size(i) > 1)
            ? 0
            : static_cast<int>(raw_strides[i]);
  }
  return strides;
}

}  // namespace

template <typename T>
class MusaDivNoNanOp : public MusaOpKernel {
 public:
  explicit MusaDivNoNanOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_0 = ctx->input(0);
    const Tensor& input_1 = ctx->input(1);

    BCast bcast(BCast::FromShape(input_0.shape()),
                BCast::FromShape(input_1.shape()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_0.shape().DebugString(),
                    " vs. ", input_1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (input_0.NumElements() == 0 || input_1.NumElements() == 0 ||
        output->NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    mTensor t_input_0 = CreateMTensor(input_0, format_);
    mTensor t_input_1 = CreateMTensor(input_1, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    ::musa::dnn::Binary op;
    auto status = op.SetMode(::musa::dnn::Binary::Mode::DIVNONAN);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA DivNoNan SetMode failed. Status code: ",
                                 static_cast<int>(status)));

    status = op.Run(handle, t_output, t_input_0, t_input_1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA DivNoNan execution failed. Status code: ",
                                 static_cast<int>(status)));
  }
};

template <>
void MusaDivNoNanOp<double>::Compute(OpKernelContext* ctx) {
  const Tensor& input_0 = ctx->input(0);
  const Tensor& input_1 = ctx->input(1);

  BCast bcast(BCast::FromShape(input_0.shape()),
              BCast::FromShape(input_1.shape()));

  OP_REQUIRES(ctx, bcast.IsValid(),
              errors::InvalidArgument("Incompatible shapes: ",
                                      input_0.shape().DebugString(), " vs. ",
                                      input_1.shape().DebugString()));

  TensorShape output_shape = BCast::ToShape(bcast.output_shape());

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  if (input_0.NumElements() == 0 || input_1.NumElements() == 0 ||
      output->NumElements() == 0) {
    return;
  }

  TensorShape padded_output = PadShapeTo4D(output_shape);
  TensorShape padded_input_0 = PadShapeTo4D(input_0.shape());
  TensorShape padded_input_1 = PadShapeTo4D(input_1.shape());

  DivNoNanDims dims;
  dims.d0 = padded_output.dim_size(0);
  dims.d1 = padded_output.dim_size(1);
  dims.d2 = padded_output.dim_size(2);
  dims.d3 = padded_output.dim_size(3);

  DivNoNanStrides input_0_strides =
      CalcDivNoNanStrides(padded_input_0, padded_output);
  DivNoNanStrides input_1_strides =
      CalcDivNoNanStrides(padded_input_1, padded_output);

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  OP_REQUIRES(ctx, stream != nullptr,
              errors::Internal("MUSA DivNoNan failed to get stream."));

  (void)musaGetLastError();
  LaunchDivNoNan<double>(
      input_0.flat<double>().data(), input_1.flat<double>().data(),
      output->flat<double>().data(), input_0_strides, input_1_strides, dims,
      output->NumElements(), stream);

  musaError_t status = musaGetLastError();
  OP_REQUIRES(ctx, status == musaSuccess,
              errors::Internal("MUSA DivNoNan custom double launch failed: ",
                               musaGetErrorString(status)));
}

#define REGISTER_MUSA_DIV_NO_NAN(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DivNoNan").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaDivNoNanOp<TYPE>);

REGISTER_MUSA_DIV_NO_NAN(float);
REGISTER_MUSA_DIV_NO_NAN(double);
REGISTER_MUSA_DIV_NO_NAN(Eigen::half);
REGISTER_MUSA_DIV_NO_NAN(bfloat16);

#undef REGISTER_MUSA_DIV_NO_NAN

}  // namespace musa
}  // namespace tensorflow
