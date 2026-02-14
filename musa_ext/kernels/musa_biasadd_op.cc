#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaBiasAddOp : public MusaOpKernel {
 public:
  explicit MusaBiasAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& bias = ctx->input(1);

    if (input.NumElements() == 0 || bias.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
      return;
    }

    int channel_dim = (data_format_ == FORMAT_NCHW) ? 1 : input.dims() - 1;
    OP_REQUIRES(ctx, bias.dim_size(0) == input.dim_size(channel_dim),
                errors::InvalidArgument("Dimension mismatch in BiasAdd"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_in = CreateMTensor(input, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    int dims_cnt = input.dims();
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);

    b_dims[channel_dim] = bias.dim_size(0);
    b_strides[channel_dim] = 1;

    mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    mBinary op;
    op.SetMode(::musa::dnn::Binary::Mode::ADD);
    auto status = op.Run(handle, mt_out, mt_in, mt_bias);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA BiasAdd failed. Status: ", (int)status));
  }

 private:
  TensorFormat data_format_;
};

#define REGISTER_MUSA_BIAS_ADD(TYPE)                              \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BiasAdd").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      MusaBiasAddOp<TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BiasAddV1").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaBiasAddOp<TYPE>);

REGISTER_MUSA_BIAS_ADD(float);
REGISTER_MUSA_BIAS_ADD(double);
REGISTER_MUSA_BIAS_ADD(Eigen::half);
REGISTER_MUSA_BIAS_ADD(bfloat16);
REGISTER_MUSA_BIAS_ADD(int32);
REGISTER_MUSA_BIAS_ADD(int64);

}  // namespace musa
}  // namespace tensorflow
