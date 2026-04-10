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

  return ::tsl::OkStatus();
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
    return ::tsl::OkStatus();
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
    return ::tsl::OkStatus();
  }

  return errors::InvalidArgument(
      "MUSA Conv2D currently only supports "
      "padding in {SAME, VALID}.");
}

template <typename T>
Status RunMusaConv2DBackpropInput(OpKernelContext* ctx,
                                  const Tensor& out_backprop,
                                  const Tensor& filter, Tensor* in_backprop,
                                  TensorFormat data_format, int stride_h,
                                  int stride_w, int dilation_h, int dilation_w,
                                  int pad_top, int pad_left) {
  auto& handle = GetHandleByCtx(ctx);

  handle.SetAllowTF32(false);
  if (data_format != FORMAT_NHWC) {
    return errors::InvalidArgument(
        "RunMusaConv2DBackpropInput currently expects NHWC tensors.");
  }

  // muDNN backward data convolution (gradient w.r.t. input)
  // out_backprop is the gradient from the next layer (dy)
  // filter is the weight tensor (w)
  // in_backprop is the gradient to propagate to the previous layer (dx)
  mTensor dy = CreateMTensor(out_backprop, mFormat::NHWC);
  mTensor dx = CreateMTensor(*in_backprop, mFormat::NHWC);
  mTensor w = CreateMTensor(filter, mFormat::HWCN);

  mConvolution conv;
  int pads[2] = {pad_top, pad_left};
  int strides[2] = {stride_h, stride_w};
  int dilations[2] = {dilation_h, dilation_w};
  mStatus status = conv.SetNdInfo(2, pads, strides, dilations);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetNdInfo failed in BackpropInput. status=",
        static_cast<int>(status));
  }

  const auto compute_mode =
      (out_backprop.dtype() == DT_FLOAT || out_backprop.dtype() == DT_DOUBLE)
          ? mConvolution::ComputeMode::SCALAR
          : mConvolution::ComputeMode::TENSOR;
  status = conv.SetComputeMode(compute_mode);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetComputeMode failed in BackpropInput. status=",
        static_cast<int>(status));
  }

  status = conv.SetGroups(1);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetGroups failed in BackpropInput. status=",
        static_cast<int>(status));
  }

  mConvolution::AlgorithmBwdData algo;
  status = conv.GetRecommendBackwardDataAlgorithm(handle, algo, dx, dy, w);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetRecommendBackwardDataAlgorithm failed. status=",
        static_cast<int>(status), ", data_format=NHWC",
        ", out_backprop_shape=", out_backprop.shape().DebugString(),
        ", filter_shape=", filter.shape().DebugString(),
        ", in_backprop_shape=", in_backprop->shape().DebugString());
  }

  size_t workspace_size = 0;
  status = conv.GetBackwardDataWorkspaceSize(handle, workspace_size, dx, dy, w,
                                             algo);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetBackwardDataWorkspaceSize failed. status=",
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
    if (bytes == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    if (bytes <= workspace_size) {
      return ::musa::dnn::MemoryHandler(workspace.flat<uint8_t>().data(),
                                        [](void*) {});
    }

    void* dynamic_ptr = nullptr;
    if (MusaAllocate(bytes, &dynamic_ptr) != mStatus::SUCCESS ||
        dynamic_ptr == nullptr) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    return ::musa::dnn::MemoryHandler(dynamic_ptr,
                                      [](void* p) { MusaFree(p); });
  };

  status = conv.RunBwdData(handle, dx, dy, w, algo, maintainer);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::RunBackwardData failed. status=",
        static_cast<int>(status));
  }

  return ::tsl::OkStatus();
}

template <typename T>
Status RunMusaConv2DBackpropFilter(OpKernelContext* ctx,
                                   const Tensor& out_backprop,
                                   const Tensor& input, Tensor* filter_backprop,
                                   TensorFormat data_format, int stride_h,
                                   int stride_w, int dilation_h, int dilation_w,
                                   int pad_top, int pad_left) {
  auto& handle = GetHandleByCtx(ctx);

  handle.SetAllowTF32(false);
  if (data_format != FORMAT_NHWC) {
    return errors::InvalidArgument(
        "RunMusaConv2DBackpropFilter currently expects NHWC tensors.");
  }

  // muDNN backward filter convolution (gradient w.r.t. filter)
  // out_backprop is the gradient from the next layer (dy)
  // input is the original input tensor (x)
  // filter_backprop is the gradient for the filter (dw)
  mTensor dy = CreateMTensor(out_backprop, mFormat::NHWC);
  mTensor x = CreateMTensor(input, mFormat::NHWC);
  mTensor dw = CreateMTensor(*filter_backprop, mFormat::HWCN);

  mConvolution conv;
  int pads[2] = {pad_top, pad_left};
  int strides[2] = {stride_h, stride_w};
  int dilations[2] = {dilation_h, dilation_w};
  mStatus status = conv.SetNdInfo(2, pads, strides, dilations);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetNdInfo failed in BackpropFilter. status=",
        static_cast<int>(status));
  }

  const auto compute_mode =
      (out_backprop.dtype() == DT_FLOAT || out_backprop.dtype() == DT_DOUBLE)
          ? mConvolution::ComputeMode::SCALAR
          : mConvolution::ComputeMode::TENSOR;
  status = conv.SetComputeMode(compute_mode);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetComputeMode failed in BackpropFilter. status=",
        static_cast<int>(status));
  }

  status = conv.SetGroups(1);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::SetGroups failed in BackpropFilter. status=",
        static_cast<int>(status));
  }

  mConvolution::AlgorithmBwdFilter algo;
  status = conv.GetRecommendBackwardFilterAlgorithm(handle, algo, dw, x, dy);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetRecommendBackwardFilterAlgorithm failed. "
        "status=",
        static_cast<int>(status), ", data_format=NHWC",
        ", out_backprop_shape=", out_backprop.shape().DebugString(),
        ", input_shape=", input.shape().DebugString(),
        ", filter_backprop_shape=", filter_backprop->shape().DebugString());
  }

  size_t workspace_size = 0;
  status = conv.GetBackwardFilterWorkspaceSize(handle, workspace_size, dw, x,
                                               dy, algo);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetBackwardFilterWorkspaceSize failed. status=",
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
    if (bytes == 0) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    if (bytes <= workspace_size) {
      return ::musa::dnn::MemoryHandler(workspace.flat<uint8_t>().data(),
                                        [](void*) {});
    }

    void* dynamic_ptr = nullptr;
    if (MusaAllocate(bytes, &dynamic_ptr) != mStatus::SUCCESS ||
        dynamic_ptr == nullptr) {
      return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    return ::musa::dnn::MemoryHandler(dynamic_ptr,
                                      [](void* p) { MusaFree(p); });
  };

  status = conv.RunBwdFilter(handle, dw, x, dy, algo, maintainer);
  if (status != mStatus::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::RunBackwardFilter failed. status=",
        static_cast<int>(status));
  }

  return ::tsl::OkStatus();
}

}  // namespace

// Conv2DBackpropInput: Compute gradient w.r.t. input
// Inputs:
//   0: input_sizes (shape of the original input)
//   1: filter (weight tensor)
//   2: out_backprop (gradient from upstream)
template <typename T>
class MusaConv2DBackpropInputOp : public MusaOpKernel {
 public:
  explicit MusaConv2DBackpropInputOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
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

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_sizes = ctx->input(0);
    const Tensor& filter = ctx->input(1);
    const Tensor& out_backprop = ctx->input(2);

    OP_REQUIRES(ctx, input_sizes.dims() == 1,
                errors::InvalidArgument("input_sizes must be 1-D, got shape: ",
                                        input_sizes.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 4,
                errors::InvalidArgument("Conv2D filter must be rank 4, got: ",
                                        filter.shape().DebugString()));
    OP_REQUIRES(ctx, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be rank 4, got: ",
                                        out_backprop.shape().DebugString()));

    // Parse input_sizes to get the original input shape
    auto input_sizes_vec = input_sizes.vec<int32>();
    const int64_t batch = input_sizes_vec(0);
    int64_t in_h, in_w, in_c;
    if (data_format_ == FORMAT_NHWC) {
      in_h = input_sizes_vec(1);
      in_w = input_sizes_vec(2);
      in_c = input_sizes_vec(3);
    } else {
      in_c = input_sizes_vec(1);
      in_h = input_sizes_vec(2);
      in_w = input_sizes_vec(3);
    }

    // TF Conv2D filter layout: [KH, KW, IC, OC] (HWIO)
    const int64_t filter_h = filter.dim_size(0);
    const int64_t filter_w = filter.dim_size(1);
    const int64_t filter_ic = filter.dim_size(2);
    const int64_t out_c = filter.dim_size(3);

    OP_REQUIRES(ctx, in_c == filter_ic,
                errors::InvalidArgument("Conv2D channel mismatch: input C=",
                                        in_c, ", filter IC=", filter_ic));

    // Compute output (forward) dimensions and padding
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

    OP_REQUIRES(ctx, pad_top == pad_bottom && pad_left == pad_right,
                errors::Unimplemented("Current MUSA Conv2D path only supports "
                                      "symmetric padding. got [top,bottom,left,"
                                      "right]=",
                                      pad_top, ",", pad_bottom, ",", pad_left,
                                      ",", pad_right));

    // Verify out_backprop shape matches expected output shape
    const int n_idx = GetTensorDimIndex(data_format_, 'N');
    const int h_idx = GetTensorDimIndex(data_format_, 'H');
    const int w_idx = GetTensorDimIndex(data_format_, 'W');
    const int c_idx = GetTensorDimIndex(data_format_, 'C');

    OP_REQUIRES(ctx, out_backprop.dim_size(n_idx) == batch,
                errors::InvalidArgument("out_backprop batch mismatch."));
    OP_REQUIRES(
        ctx, out_backprop.dim_size(h_idx) == out_h,
        errors::InvalidArgument("out_backprop height mismatch: expected ",
                                out_h, ", got ", out_backprop.dim_size(h_idx)));
    OP_REQUIRES(
        ctx, out_backprop.dim_size(w_idx) == out_w,
        errors::InvalidArgument("out_backprop width mismatch: expected ", out_w,
                                ", got ", out_backprop.dim_size(w_idx)));
    OP_REQUIRES(
        ctx, out_backprop.dim_size(c_idx) == out_c,
        errors::InvalidArgument("out_backprop channel mismatch: expected ",
                                out_c, ", got ", out_backprop.dim_size(c_idx)));

    // Allocate output tensor for input gradient
    TensorShape in_backprop_shape;
    if (data_format_ == FORMAT_NHWC) {
      in_backprop_shape = TensorShape({batch, in_h, in_w, in_c});
    } else {
      in_backprop_shape = TensorShape({batch, in_c, in_h, in_w});
    }

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, in_backprop_shape, &in_backprop));
    if (in_backprop->NumElements() == 0) {
      return;
    }

    if (data_format_ == FORMAT_NHWC) {
      OP_REQUIRES_OK(ctx, RunMusaConv2DBackpropInput<T>(
                              ctx, out_backprop, filter, in_backprop,
                              FORMAT_NHWC, stride_h_, stride_w_, dilation_h_,
                              dilation_w_, pad_top, pad_left));
      return;
    }

    // NCHW fallback: transpose to NHWC, compute, transpose back
    Tensor out_backprop_nhwc;
    Tensor in_backprop_nhwc;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(out_backprop.dtype(),
                                      TensorShape({batch, out_h, out_w, out_c}),
                                      &out_backprop_nhwc));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(in_backprop->dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &in_backprop_nhwc));

    static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
    static const std::vector<int64_t> kPermNhwcToNchw = {0, 3, 1, 2};
    OP_REQUIRES_OK(ctx,
                   PermuteTensorOnMusa(ctx, out_backprop, &out_backprop_nhwc,
                                       kPermNchwToNhwc));
    OP_REQUIRES_OK(ctx, RunMusaConv2DBackpropInput<T>(
                            ctx, out_backprop_nhwc, filter, &in_backprop_nhwc,
                            FORMAT_NHWC, stride_h_, stride_w_, dilation_h_,
                            dilation_w_, pad_top, pad_left));
    OP_REQUIRES_OK(ctx, PermuteTensorOnMusa(ctx, in_backprop_nhwc, in_backprop,
                                            kPermNhwcToNchw));
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

// Conv2DBackpropFilter: Compute gradient w.r.t. filter
// Inputs:
//   0: input (original input tensor)
//   1: filter_sizes (shape of the filter)
//   2: out_backprop (gradient from upstream)
template <typename T>
class MusaConv2DBackpropFilterOp : public MusaOpKernel {
 public:
  explicit MusaConv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
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

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& filter_sizes = ctx->input(1);
    const Tensor& out_backprop = ctx->input(2);

    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("Conv2D input must be rank 4, got: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, filter_sizes.dims() == 1,
                errors::InvalidArgument("filter_sizes must be 1-D, got shape: ",
                                        filter_sizes.shape().DebugString()));
    OP_REQUIRES(ctx, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be rank 4, got: ",
                                        out_backprop.shape().DebugString()));

    // Parse filter_sizes to get the filter shape
    auto filter_sizes_vec = filter_sizes.vec<int32>();
    const int64_t filter_h = filter_sizes_vec(0);
    const int64_t filter_w = filter_sizes_vec(1);
    const int64_t filter_ic = filter_sizes_vec(2);
    const int64_t filter_oc = filter_sizes_vec(3);

    // Get input dimensions
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

    // Compute output (forward) dimensions and padding
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

    OP_REQUIRES(ctx, pad_top == pad_bottom && pad_left == pad_right,
                errors::Unimplemented("Current MUSA Conv2D path only supports "
                                      "symmetric padding. got [top,bottom,left,"
                                      "right]=",
                                      pad_top, ",", pad_bottom, ",", pad_left,
                                      ",", pad_right));

    // Verify out_backprop shape matches expected output shape
    OP_REQUIRES(ctx, out_backprop.dim_size(n_idx) == batch,
                errors::InvalidArgument("out_backprop batch mismatch."));
    OP_REQUIRES(
        ctx, out_backprop.dim_size(h_idx) == out_h,
        errors::InvalidArgument("out_backprop height mismatch: expected ",
                                out_h, ", got ", out_backprop.dim_size(h_idx)));
    OP_REQUIRES(
        ctx, out_backprop.dim_size(w_idx) == out_w,
        errors::InvalidArgument("out_backprop width mismatch: expected ", out_w,
                                ", got ", out_backprop.dim_size(w_idx)));
    OP_REQUIRES(ctx, out_backprop.dim_size(c_idx) == filter_oc,
                errors::InvalidArgument(
                    "out_backprop channel mismatch: expected ", filter_oc,
                    ", got ", out_backprop.dim_size(c_idx)));

    // Allocate output tensor for filter gradient
    TensorShape filter_backprop_shape =
        TensorShape({filter_h, filter_w, filter_ic, filter_oc});

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, filter_backprop_shape, &filter_backprop));
    if (filter_backprop->NumElements() == 0) {
      return;
    }

    if (data_format_ == FORMAT_NHWC) {
      OP_REQUIRES_OK(ctx, RunMusaConv2DBackpropFilter<T>(
                              ctx, out_backprop, input, filter_backprop,
                              FORMAT_NHWC, stride_h_, stride_w_, dilation_h_,
                              dilation_w_, pad_top, pad_left));
      return;
    }

    // NCHW fallback: transpose to NHWC, compute, transpose back
    Tensor input_nhwc;
    Tensor out_backprop_nhwc;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(input.dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &input_nhwc));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(out_backprop.dtype(),
                                TensorShape({batch, out_h, out_w, filter_oc}),
                                &out_backprop_nhwc));

    static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusa(ctx, input, &input_nhwc, kPermNchwToNhwc));
    OP_REQUIRES_OK(ctx,
                   PermuteTensorOnMusa(ctx, out_backprop, &out_backprop_nhwc,
                                       kPermNchwToNhwc));
    OP_REQUIRES_OK(ctx, RunMusaConv2DBackpropFilter<T>(
                            ctx, out_backprop_nhwc, input_nhwc, filter_backprop,
                            FORMAT_NHWC, stride_h_, stride_w_, dilation_h_,
                            dilation_w_, pad_top, pad_left));
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

#define REGISTER_MUSA_CONV2D_BACKPROP_INPUT(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("Conv2DBackpropInput").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaConv2DBackpropInputOp<TYPE>)

#define REGISTER_MUSA_CONV2D_BACKPROP_FILTER(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv2DBackpropFilter").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaConv2DBackpropFilterOp<TYPE>)

REGISTER_MUSA_CONV2D_BACKPROP_INPUT(float);
REGISTER_MUSA_CONV2D_BACKPROP_INPUT(Eigen::half);
REGISTER_MUSA_CONV2D_BACKPROP_INPUT(bfloat16);

REGISTER_MUSA_CONV2D_BACKPROP_FILTER(float);
REGISTER_MUSA_CONV2D_BACKPROP_FILTER(Eigen::half);
REGISTER_MUSA_CONV2D_BACKPROP_FILTER(bfloat16);

#undef REGISTER_MUSA_CONV2D_BACKPROP_INPUT
#undef REGISTER_MUSA_CONV2D_BACKPROP_FILTER

}  // namespace musa
}  // namespace tensorflow
