#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

inline int GetDimFromAttr(const std::vector<int32>& attr, TensorFormat format,
                          char dim) {
  const int index = GetTensorDimIndex(format, dim);
  return (index >= 0) ? attr[index] : -1;
}

Status PermuteTensorOnMusa(OpKernelContext* ctx, const Tensor& input,
                           Tensor* output, const std::vector<int64_t>& perm) {
  if (input.dims() != static_cast<int>(perm.size())) {
    return errors::InvalidArgument("Permute rank mismatch. input_rank=",
                                   input.dims(), ", perm_size=", perm.size());
  }

  auto& handle = GetHandleByCtx(ctx);

  mTensor in_mt = CreateMTensor(input);
  mTensor out_mt = CreateMTensor(*output);

  mPermute permute_op;
  mStatus status = permute_op.ConfigDimStride(
      out_mt, in_mt, static_cast<int>(perm.size()), perm.data());
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Permute::ConfigDimStride failed. status=",
                            static_cast<int>(status));
  }

  status = permute_op.Run(handle, out_mt, in_mt);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Permute::Run failed. status=",
                            static_cast<int>(status));
  }

  return Status::OK();
}

Status ComputeOutputAndPadding2D(int64_t in_h, int64_t in_w, int64_t window_h,
                                 int64_t window_w, int stride_h, int stride_w,
                                 Padding padding, int64_t* out_h,
                                 int64_t* out_w, int* pad_top, int* pad_bottom,
                                 int* pad_left, int* pad_right) {
  if (padding == Padding::VALID) {
    *out_h = std::max<int64_t>(0, (in_h - window_h + stride_h) / stride_h);
    *out_w = std::max<int64_t>(0, (in_w - window_w + stride_w) / stride_w);
    *pad_top = 0;
    *pad_bottom = 0;
    *pad_left = 0;
    *pad_right = 0;
    return Status::OK();
  }

  if (padding == Padding::SAME) {
    *out_h = (in_h + stride_h - 1) / stride_h;
    *out_w = (in_w + stride_w - 1) / stride_w;

    const int64_t pad_h =
        std::max<int64_t>(0, (*out_h - 1) * stride_h + window_h - in_h);
    const int64_t pad_w =
        std::max<int64_t>(0, (*out_w - 1) * stride_w + window_w - in_w);

    *pad_top = static_cast<int>(pad_h / 2);
    *pad_bottom = static_cast<int>(pad_h - *pad_top);
    *pad_left = static_cast<int>(pad_w / 2);
    *pad_right = static_cast<int>(pad_w - *pad_left);
    return Status::OK();
  }

  return errors::InvalidArgument(
      "MUSA MaxPool currently only supports "
      "padding in {SAME, VALID}.");
}

template <typename T>
Status RunMusaMaxPool(OpKernelContext* ctx, const Tensor& input, Tensor* output,
                      TensorFormat data_format, int window_h, int window_w,
                      int stride_h, int stride_w, int pad_top, int pad_left) {
  auto& handle = GetHandleByCtx(ctx);

  mTensor x = CreateMTensor(input, mFormat::NHWC);
  mTensor y = CreateMTensor(*output, mFormat::NHWC);

  mPooling pool;
  pool.SetMode(::musa::dnn::Pooling::Mode::MAXPOOL);

  int pads[2] = {pad_top, pad_left};
  int window[2] = {window_h, window_w};
  int strides[2] = {stride_h, stride_w};
  int dilation[2] = {1, 1};

  mStatus status = pool.SetNdInfo(2, window, pads, strides, dilation);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Pooling::SetNdInfo failed. status=",
                            static_cast<int>(status));
  }

  mTensor indices;
  status = pool.Run(handle, y, x, indices);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Pooling::Run failed. status=",
                            static_cast<int>(status));
  }

  return Status::OK();
}

}  // namespace

template <typename T>
class MusaMaxPoolOp : public MusaOpKernel {
 public:
  explicit MusaMaxPoolOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str_));

    OP_REQUIRES(ctx, FormatFromString(data_format_str_, &data_format_),
                errors::InvalidArgument("Invalid MaxPool data_format: ",
                                        data_format_str_));
    OP_REQUIRES(
        ctx, data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW,
        errors::InvalidArgument("MaxPool only supports NHWC/NCHW, got: ",
                                data_format_str_));

    OP_REQUIRES_OK(ctx, GetPaddingFromString(padding_str_, &padding_));
    OP_REQUIRES(
        ctx, padding_ == Padding::SAME || padding_ == Padding::VALID,
        errors::InvalidArgument("MaxPool only supports SAME/VALID padding."));

    OP_REQUIRES(
        ctx, ksize_.size() == 4,
        errors::InvalidArgument("MaxPool ksize attr must have 4 elements."));
    OP_REQUIRES(
        ctx, strides_.size() == 4,
        errors::InvalidArgument("MaxPool strides attr must have 4 elements."));

    const int ksize_n = GetDimFromAttr(ksize_, data_format_, 'N');
    const int ksize_c = GetDimFromAttr(ksize_, data_format_, 'C');
    const int stride_n = GetDimFromAttr(strides_, data_format_, 'N');
    const int stride_c = GetDimFromAttr(strides_, data_format_, 'C');

    window_h_ = GetDimFromAttr(ksize_, data_format_, 'H');
    window_w_ = GetDimFromAttr(ksize_, data_format_, 'W');
    stride_h_ = GetDimFromAttr(strides_, data_format_, 'H');
    stride_w_ = GetDimFromAttr(strides_, data_format_, 'W');

    OP_REQUIRES(ctx, ksize_n == 1 && ksize_c == 1,
                errors::InvalidArgument("MaxPool does not support pooling on "
                                        "batch/channel dims."));
    OP_REQUIRES(ctx, stride_n == 1 && stride_c == 1,
                errors::InvalidArgument("MaxPool does not support strides on "
                                        "batch/channel dims."));
    OP_REQUIRES(
        ctx, window_h_ > 0 && window_w_ > 0,
        errors::InvalidArgument("MaxPool spatial window sizes must be > 0."));
    OP_REQUIRES(
        ctx, stride_h_ > 0 && stride_w_ > 0,
        errors::InvalidArgument("MaxPool spatial strides must be > 0."));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("MaxPool input must be rank 4, got: ",
                                        input.shape().DebugString()));

    const int n_idx = GetTensorDimIndex(data_format_, 'N');
    const int h_idx = GetTensorDimIndex(data_format_, 'H');
    const int w_idx = GetTensorDimIndex(data_format_, 'W');
    const int c_idx = GetTensorDimIndex(data_format_, 'C');

    const int64_t batch = input.dim_size(n_idx);
    const int64_t in_h = input.dim_size(h_idx);
    const int64_t in_w = input.dim_size(w_idx);
    const int64_t in_c = input.dim_size(c_idx);

    int64_t out_h = 0;
    int64_t out_w = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    OP_REQUIRES_OK(ctx, ComputeOutputAndPadding2D(
                            in_h, in_w, window_h_, window_w_, stride_h_,
                            stride_w_, padding_, &out_h, &out_w, &pad_top,
                            &pad_bottom, &pad_left, &pad_right));

    // Current muDNN SetNdInfo uses symmetric pad value per spatial dimension.
    OP_REQUIRES(ctx, pad_top == pad_bottom && pad_left == pad_right,
                errors::Unimplemented("Current MUSA MaxPool path only supports "
                                      "symmetric padding. got [top,bottom,left,"
                                      "right]=",
                                      pad_top, ",", pad_bottom, ",", pad_left,
                                      ",", pad_right));

    TensorShape output_shape;
    if (data_format_ == FORMAT_NHWC) {
      output_shape = TensorShape({batch, out_h, out_w, in_c});
    } else {
      output_shape = TensorShape({batch, in_c, out_h, out_w});
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    if (data_format_ == FORMAT_NHWC) {
      OP_REQUIRES_OK(ctx, RunMusaMaxPool<T>(ctx, input, output, FORMAT_NHWC,
                                            window_h_, window_w_, stride_h_,
                                            stride_w_, pad_top, pad_left));
      return;
    }

    // For NCHW: transpose to NHWC, run NHWC pool, transpose back
    Tensor input_nhwc;
    Tensor output_nhwc;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(input.dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &input_nhwc));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(output->dtype(),
                                      TensorShape({batch, out_h, out_w, in_c}),
                                      &output_nhwc));

    static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
    static const std::vector<int64_t> kPermNhwcToNchw = {0, 3, 1, 2};
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusa(ctx, input, &input_nhwc, kPermNchwToNhwc));
    OP_REQUIRES_OK(
        ctx,
        RunMusaMaxPool<T>(ctx, input_nhwc, &output_nhwc, FORMAT_NHWC, window_h_,
                          window_w_, stride_h_, stride_w_, pad_top, pad_left));
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusa(ctx, output_nhwc, output, kPermNhwcToNchw));
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> strides_;
  std::string padding_str_;
  std::string data_format_str_;

  TensorFormat data_format_ = FORMAT_NHWC;
  Padding padding_ = Padding::SAME;
  int window_h_ = 1;
  int window_w_ = 1;
  int stride_h_ = 1;
  int stride_w_ = 1;
};

#define REGISTER_MUSA_MAXPOOL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("MaxPool").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMaxPoolOp<TYPE>)

REGISTER_MUSA_MAXPOOL(float);
REGISTER_MUSA_MAXPOOL(Eigen::half);
REGISTER_MUSA_MAXPOOL(bfloat16);

#undef REGISTER_MUSA_MAXPOOL

}  // namespace musa
}  // namespace tensorflow