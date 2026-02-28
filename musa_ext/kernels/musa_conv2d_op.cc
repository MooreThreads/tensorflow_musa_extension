#include <algorithm>
#include <cstdint>
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

  // Use raw ND descriptors for permutation to avoid coupling to a specific
  // 4-D layout semantic (NHWC/NCHW/HWCN).
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

Status ComputeOutputAndPadding2D(int64_t in_h, int64_t in_w, int64_t filter_h,
                                 int64_t filter_w, int stride_h, int stride_w,
                                 int dilation_h, int dilation_w,
                                 Padding padding, int64_t* out_h,
                                 int64_t* out_w, int* pad_top, int* pad_bottom,
                                 int* pad_left, int* pad_right) {
  const int64_t effective_kh = (filter_h - 1) * dilation_h + 1;
  const int64_t effective_kw = (filter_w - 1) * dilation_w + 1;

  if (padding == Padding::VALID) {
    *out_h = std::max<int64_t>(0, (in_h + stride_h - effective_kh) / stride_h);
    *out_w = std::max<int64_t>(0, (in_w + stride_w - effective_kw) / stride_w);
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
        std::max<int64_t>(0, (*out_h - 1) * stride_h + effective_kh - in_h);
    const int64_t pad_w =
        std::max<int64_t>(0, (*out_w - 1) * stride_w + effective_kw - in_w);

    *pad_top = static_cast<int>(pad_h / 2);
    *pad_bottom = static_cast<int>(pad_h - *pad_top);
    *pad_left = static_cast<int>(pad_w / 2);
    *pad_right = static_cast<int>(pad_w - *pad_left);
    return Status::OK();
  }

  return errors::InvalidArgument(
      "MUSA Conv2D currently only supports "
      "padding in {SAME, VALID}.");
}

template <typename T>
Status RunMusaConv2D(OpKernelContext* ctx, const Tensor& input,
                     const Tensor& filter, Tensor* output,
                     TensorFormat data_format, int stride_h, int stride_w,
                     int dilation_h, int dilation_w, int pad_top,
                     int pad_left) {
  auto& handle = GetHandleByCtx(ctx);

  // Match CPU float32 numerics as much as possible.
  handle.SetAllowTF32(false);
  if (data_format != FORMAT_NHWC) {
    return errors::InvalidArgument(
        "RunMusaConv2D currently expects NHWC tensors.");
  }

  // muDNN NHWC forward supports filter format HWCN.
  // TF Conv2D filter layout HWIO is compatible with HWCN semantics here.
  mTensor x = CreateMTensor(input, mFormat::NHWC);
  mTensor y = CreateMTensor(*output, mFormat::NHWC);
  mTensor w = CreateMTensor(filter, mFormat::HWCN);

  mConvolution conv;
  int pads[2] = {pad_top, pad_left};
  int strides[2] = {stride_h, stride_w};
  int dilations[2] = {dilation_h, dilation_w};
  mStatus status = conv.SetNdInfo(2, pads, strides, dilations);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Convolution::SetNdInfo failed. status=",
                            static_cast<int>(status));
  }

  const auto compute_mode =
      (input.dtype() == DT_FLOAT || input.dtype() == DT_DOUBLE)
          ? mConvolution::ComputeMode::SCALAR
          : mConvolution::ComputeMode::TENSOR;
  status = conv.SetComputeMode(compute_mode);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Convolution::SetComputeMode failed. status=",
                            static_cast<int>(status));
  }

  status = conv.SetGroups(1);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Convolution::SetGroups failed. status=",
                            static_cast<int>(status));
  }

  mConvolution::Algorithm algo;
  status = conv.GetRecommendForwardAlgorithm(handle, algo, y, x, w);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetRecommendForwardAlgorithm failed. status=",
        static_cast<int>(status), ", data_format=NHWC",
        ", input_shape=", input.shape().DebugString(),
        ", filter_shape=", filter.shape().DebugString(),
        ", output_shape=", output->shape().DebugString());
  }

  size_t workspace_size = 0;
  status = conv.GetForwardWorkspaceSize(handle, workspace_size, y, x, w, algo);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetForwardWorkspaceSize failed. status=",
        static_cast<int>(status));
  }

  Tensor workspace;
  if (workspace_size > 0) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_UINT8, TensorShape({static_cast<int64_t>(workspace_size)}),
        &workspace));
  }

  ::musa::dnn::MemoryMaintainer maintainer =
      [&workspace, workspace_size](size_t bytes) -> ::musa::dnn::MemoryHandler {
    if (bytes == 0 || bytes > workspace_size) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    return ::musa::dnn::MemoryHandler(workspace.flat<uint8_t>().data(),
                                      [](void*) {});
  };

  status = conv.Run(handle, y, x, w, algo, maintainer);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Convolution::Run failed. status=",
                            static_cast<int>(status));
  }

  return Status::OK();
}

}  // namespace

template <typename T>
class MusaConv2DOp : public MusaOpKernel {
 public:
  explicit MusaConv2DOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str_));

    OP_REQUIRES(ctx, FormatFromString(data_format_str_, &data_format_),
                errors::InvalidArgument("Invalid Conv2D data_format: ",
                                        data_format_str_));
    OP_REQUIRES(ctx, data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW,
                errors::InvalidArgument("Conv2D only supports NHWC/NCHW, got: ",
                                        data_format_str_));

    OP_REQUIRES_OK(ctx, GetPaddingFromString(padding_str_, &padding_));
    OP_REQUIRES(
        ctx, padding_ == Padding::SAME || padding_ == Padding::VALID,
        errors::InvalidArgument("Conv2D only supports SAME/VALID padding."));

    OP_REQUIRES(
        ctx, strides_.size() == 4,
        errors::InvalidArgument("Conv2D strides attr must have 4 elements."));
    OP_REQUIRES(
        ctx, dilations_.size() == 4,
        errors::InvalidArgument("Conv2D dilations attr must have 4 elements."));

    const int stride_n = GetDimFromAttr(strides_, data_format_, 'N');
    const int stride_c = GetDimFromAttr(strides_, data_format_, 'C');
    const int dilation_n = GetDimFromAttr(dilations_, data_format_, 'N');
    const int dilation_c = GetDimFromAttr(dilations_, data_format_, 'C');

    stride_h_ = GetDimFromAttr(strides_, data_format_, 'H');
    stride_w_ = GetDimFromAttr(strides_, data_format_, 'W');
    dilation_h_ = GetDimFromAttr(dilations_, data_format_, 'H');
    dilation_w_ = GetDimFromAttr(dilations_, data_format_, 'W');

    OP_REQUIRES(ctx, stride_n == 1 && stride_c == 1,
                errors::InvalidArgument("Conv2D does not support strides on "
                                        "batch/channel dims."));
    OP_REQUIRES(ctx, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument("Conv2D does not support dilations on "
                                        "batch/channel dims."));
    OP_REQUIRES(ctx, stride_h_ > 0 && stride_w_ > 0,
                errors::InvalidArgument("Conv2D spatial strides must be > 0."));
    OP_REQUIRES(
        ctx, dilation_h_ > 0 && dilation_w_ > 0,
        errors::InvalidArgument("Conv2D spatial dilations must be > 0."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& filter = ctx->input(1);

    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("Conv2D input must be rank 4, got: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 4,
                errors::InvalidArgument("Conv2D filter must be rank 4, got: ",
                                        filter.shape().DebugString()));

    // TF Conv2D filter layout: [KH, KW, IC, OC] (HWIO)
    const int64_t filter_h = filter.dim_size(0);
    const int64_t filter_w = filter.dim_size(1);
    const int64_t filter_ic = filter.dim_size(2);
    const int64_t out_c = filter.dim_size(3);

    const int n_idx = GetTensorDimIndex(data_format_, 'N');
    const int h_idx = GetTensorDimIndex(data_format_, 'H');
    const int w_idx = GetTensorDimIndex(data_format_, 'W');
    const int c_idx = GetTensorDimIndex(data_format_, 'C');

    const int64_t batch = input.dim_size(n_idx);
    const int64_t in_h = input.dim_size(h_idx);
    const int64_t in_w = input.dim_size(w_idx);
    const int64_t in_c = input.dim_size(c_idx);

    OP_REQUIRES(ctx, in_c == filter_ic,
                errors::InvalidArgument("Conv2D channel mismatch: input C=",
                                        in_c, ", filter IC=", filter_ic));

    int64_t out_h = 0;
    int64_t out_w = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    OP_REQUIRES_OK(
        ctx, ComputeOutputAndPadding2D(
                 in_h, in_w, filter_h, filter_w, stride_h_, stride_w_,
                 dilation_h_, dilation_w_, padding_, &out_h, &out_w, &pad_top,
                 &pad_bottom, &pad_left, &pad_right));

    // Current muDNN SetNdInfo uses symmetric pad value per spatial dimension.
    OP_REQUIRES(ctx, pad_top == pad_bottom && pad_left == pad_right,
                errors::Unimplemented("Current MUSA Conv2D path only supports "
                                      "symmetric padding. got [top,bottom,left,"
                                      "right]=",
                                      pad_top, ",", pad_bottom, ",", pad_left,
                                      ",", pad_right));

    TensorShape output_shape;
    if (data_format_ == FORMAT_NHWC) {
      output_shape = TensorShape({batch, out_h, out_w, out_c});
    } else {
      output_shape = TensorShape({batch, out_c, out_h, out_w});
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    if (data_format_ == FORMAT_NHWC) {
      OP_REQUIRES_OK(
          ctx, RunMusaConv2D<T>(ctx, input, filter, output, FORMAT_NHWC,
                                stride_h_, stride_w_, dilation_h_, dilation_w_,
                                pad_top, pad_left));
      return;
    }

    // Robust fallback for NCHW: transpose to NHWC, run NHWC conv, transpose
    // back. This avoids current native NCHW path instability.
    Tensor input_nhwc;
    Tensor output_nhwc;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(input.dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &input_nhwc));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(output->dtype(),
                                      TensorShape({batch, out_h, out_w, out_c}),
                                      &output_nhwc));

    static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
    static const std::vector<int64_t> kPermNhwcToNchw = {0, 3, 1, 2};
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusa(ctx, input, &input_nhwc, kPermNchwToNhwc));
    OP_REQUIRES_OK(
        ctx, RunMusaConv2D<T>(ctx, input_nhwc, filter, &output_nhwc,
                              FORMAT_NHWC, stride_h_, stride_w_, dilation_h_,
                              dilation_w_, pad_top, pad_left));
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusa(ctx, output_nhwc, output, kPermNhwcToNchw));
  }

 private:
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  std::string padding_str_;
  std::string data_format_str_;

  TensorFormat data_format_ = FORMAT_NHWC;
  Padding padding_ = Padding::SAME;
  int stride_h_ = 1;
  int stride_w_ = 1;
  int dilation_h_ = 1;
  int dilation_w_ = 1;
};

#define REGISTER_MUSA_CONV2D(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Conv2D").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaConv2DOp<TYPE>)

REGISTER_MUSA_CONV2D(float);
REGISTER_MUSA_CONV2D(Eigen::half);
REGISTER_MUSA_CONV2D(bfloat16);

#undef REGISTER_MUSA_CONV2D

}  // namespace musa
}  // namespace tensorflow
