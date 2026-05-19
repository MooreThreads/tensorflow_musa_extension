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

inline int GetDimFromAttrGrad(const std::vector<int32>& attr,
                              TensorFormat format, char dim) {
  const int index = GetTensorDimIndex(format, dim);
  return (index >= 0) ? attr[index] : -1;
}

Status PermuteTensorOnMusaGrad(OpKernelContext* ctx, const Tensor& input,
                               Tensor* output,
                               const std::vector<int64_t>& perm) {
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

  return OkStatus();
}

Status ComputeOutputAndPadding2DGrad(
    int64_t in_h, int64_t in_w, int64_t window_h, int64_t window_w,
    int stride_h, int stride_w, Padding padding, int64_t* out_h, int64_t* out_w,
    int* pad_top, int* pad_bottom, int* pad_left, int* pad_right) {
  if (padding == Padding::VALID) {
    *out_h = std::max<int64_t>(0, (in_h - window_h + stride_h) / stride_h);
    *out_w = std::max<int64_t>(0, (in_w - window_w + stride_w) / stride_w);
    *pad_top = 0;
    *pad_bottom = 0;
    *pad_left = 0;
    *pad_right = 0;
    return OkStatus();
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
    return OkStatus();
  }

  return errors::InvalidArgument(
      "MUSA MaxPoolGrad currently only supports "
      "padding in {SAME, VALID}.");
}

// Re-runs the forward MaxPool pass on orig_input (NHWC) to obtain the
// max-position indices, then propagates grad_output (NHWC) back through
// RunBwd to produce grad_input (NHWC).
template <typename T>
Status RunMusaMaxPoolGrad(OpKernelContext* ctx, const Tensor& orig_input,
                          const Tensor& grad_output, Tensor* grad_input,
                          int window_h, int window_w, int stride_h,
                          int stride_w, int pad_top, int pad_left) {
  auto& handle = GetHandleByCtx(ctx);

  // Configure pooling with the same parameters as the forward pass.
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

  // Allocate a scratch tensor for the forward output (same shape as
  // grad_output).  We need to execute the forward pass to obtain the internal
  // max-position indices that muDNN uses for the backward pass.
  Tensor fwd_output_scratch;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      grad_output.dtype(), grad_output.shape(), &fwd_output_scratch));

  // Allocate GPU memory for the max-position indices (INT64, same shape as
  // the pooling output).  muDNN requires a pre-allocated device buffer; a
  // default-constructed mTensor has no backing memory and causes RunBwd to
  // fail with INTERNAL_ERROR (status=4).
  Tensor indices_tensor;
  TF_RETURN_IF_ERROR(
      ctx->allocate_temp(DT_INT64, grad_output.shape(), &indices_tensor));

  // Build the indices mTensor descriptor manually: CreateMTensor() derives the
  // type from t.dtype(), but indices_tensor.dtype() is DT_INT64 which maps
  // to mType::INT64, so we can simply use CreateMTensor here.
  mTensor indices_mt = CreateMTensor(indices_tensor, mFormat::NHWC);

  mTensor x = CreateMTensor(orig_input, mFormat::NHWC);
  mTensor y = CreateMTensor(fwd_output_scratch, mFormat::NHWC);

  // Run the forward pass to populate the indices buffer.
  status = pool.Run(handle, y, x, indices_mt);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Pooling::Run (for indices) failed. status=",
                            static_cast<int>(status));
  }

  // Propagate gradients backward.
  mTensor dy = CreateMTensor(grad_output, mFormat::NHWC);
  mTensor dx = CreateMTensor(*grad_input, mFormat::NHWC);

  status = pool.RunBwd(handle, dx, dy, indices_mt);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("muDNN Pooling::RunBwd failed. status=",
                            static_cast<int>(status));
  }

  return OkStatus();
}

}  // namespace

template <typename T>
class MusaMaxPoolGradOp : public MusaOpKernel {
 public:
  explicit MusaMaxPoolGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str_));

    OP_REQUIRES(ctx, FormatFromString(data_format_str_, &data_format_),
                errors::InvalidArgument("Invalid MaxPoolGrad data_format: ",
                                        data_format_str_));
    OP_REQUIRES(
        ctx, data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW,
        errors::InvalidArgument("MaxPoolGrad only supports NHWC/NCHW, got: ",
                                data_format_str_));

    OP_REQUIRES_OK(ctx, GetPaddingFromString(padding_str_, &padding_));
    OP_REQUIRES(ctx, padding_ == Padding::SAME || padding_ == Padding::VALID,
                errors::InvalidArgument(
                    "MaxPoolGrad only supports SAME/VALID padding."));

    OP_REQUIRES(ctx, ksize_.size() == 4,
                errors::InvalidArgument(
                    "MaxPoolGrad ksize attr must have 4 elements."));
    OP_REQUIRES(ctx, strides_.size() == 4,
                errors::InvalidArgument(
                    "MaxPoolGrad strides attr must have 4 elements."));

    const int ksize_n = GetDimFromAttrGrad(ksize_, data_format_, 'N');
    const int ksize_c = GetDimFromAttrGrad(ksize_, data_format_, 'C');
    const int stride_n = GetDimFromAttrGrad(strides_, data_format_, 'N');
    const int stride_c = GetDimFromAttrGrad(strides_, data_format_, 'C');

    window_h_ = GetDimFromAttrGrad(ksize_, data_format_, 'H');
    window_w_ = GetDimFromAttrGrad(ksize_, data_format_, 'W');
    stride_h_ = GetDimFromAttrGrad(strides_, data_format_, 'H');
    stride_w_ = GetDimFromAttrGrad(strides_, data_format_, 'W');

    OP_REQUIRES(
        ctx, ksize_n == 1 && ksize_c == 1,
        errors::InvalidArgument(
            "MaxPoolGrad does not support pooling on batch/channel dims."));
    OP_REQUIRES(
        ctx, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument(
            "MaxPoolGrad does not support strides on batch/channel dims."));
    OP_REQUIRES(ctx, window_h_ > 0 && window_w_ > 0,
                errors::InvalidArgument(
                    "MaxPoolGrad spatial window sizes must be > 0."));
    OP_REQUIRES(
        ctx, stride_h_ > 0 && stride_w_ > 0,
        errors::InvalidArgument("MaxPoolGrad spatial strides must be > 0."));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    // Inputs:
    //   0: orig_input  – the original input tensor fed to MaxPool
    //   1: orig_output – the original output tensor of MaxPool (unused by
    //                    muDNN, required by the op signature)
    //   2: grad        – upstream gradient (same shape as orig_output)
    const Tensor& orig_input = ctx->input(0);
    const Tensor& grad = ctx->input(2);

    OP_REQUIRES(
        ctx, orig_input.dims() == 4,
        errors::InvalidArgument("MaxPoolGrad orig_input must be rank 4, got: ",
                                orig_input.shape().DebugString()));
    OP_REQUIRES(
        ctx, grad.dims() == 4,
        errors::InvalidArgument("MaxPoolGrad grad must be rank 4, got: ",
                                grad.shape().DebugString()));

    const int n_idx = GetTensorDimIndex(data_format_, 'N');
    const int h_idx = GetTensorDimIndex(data_format_, 'H');
    const int w_idx = GetTensorDimIndex(data_format_, 'W');
    const int c_idx = GetTensorDimIndex(data_format_, 'C');

    const int64_t batch = orig_input.dim_size(n_idx);
    const int64_t in_h = orig_input.dim_size(h_idx);
    const int64_t in_w = orig_input.dim_size(w_idx);
    const int64_t in_c = orig_input.dim_size(c_idx);

    // Verify that the grad shape matches what MaxPool would produce.
    int64_t out_h = 0;
    int64_t out_w = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    OP_REQUIRES_OK(ctx, ComputeOutputAndPadding2DGrad(
                            in_h, in_w, window_h_, window_w_, stride_h_,
                            stride_w_, padding_, &out_h, &out_w, &pad_top,
                            &pad_bottom, &pad_left, &pad_right));

    OP_REQUIRES(ctx, pad_top == pad_bottom && pad_left == pad_right,
                errors::Unimplemented(
                    "Current MUSA MaxPoolGrad path only supports "
                    "symmetric padding. got [top,bottom,left,right]=",
                    pad_top, ",", pad_bottom, ",", pad_left, ",", pad_right));

    // Allocate output gradient tensor with the same shape as orig_input.
    Tensor* grad_input = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, orig_input.shape(), &grad_input));

    if (grad_input->NumElements() == 0) {
      return;
    }

    if (data_format_ == FORMAT_NHWC) {
      OP_REQUIRES_OK(
          ctx, RunMusaMaxPoolGrad<T>(ctx, orig_input, grad, grad_input,
                                     window_h_, window_w_, stride_h_, stride_w_,
                                     pad_top, pad_left));
      return;
    }

    // NCHW path: transpose both orig_input and grad to NHWC, run the
    // backward in NHWC, then transpose the result back to NCHW.
    Tensor orig_input_nhwc;
    Tensor grad_nhwc;
    Tensor grad_input_nhwc;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(orig_input.dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &orig_input_nhwc));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(grad.dtype(),
                                      TensorShape({batch, out_h, out_w, in_c}),
                                      &grad_nhwc));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(grad_input->dtype(),
                                      TensorShape({batch, in_h, in_w, in_c}),
                                      &grad_input_nhwc));

    static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
    static const std::vector<int64_t> kPermNhwcToNchw = {0, 3, 1, 2};

    OP_REQUIRES_OK(ctx,
                   PermuteTensorOnMusaGrad(ctx, orig_input, &orig_input_nhwc,
                                           kPermNchwToNhwc));
    OP_REQUIRES_OK(
        ctx, PermuteTensorOnMusaGrad(ctx, grad, &grad_nhwc, kPermNchwToNhwc));
    OP_REQUIRES_OK(
        ctx, RunMusaMaxPoolGrad<T>(ctx, orig_input_nhwc, grad_nhwc,
                                   &grad_input_nhwc, window_h_, window_w_,
                                   stride_h_, stride_w_, pad_top, pad_left));
    OP_REQUIRES_OK(ctx, PermuteTensorOnMusaGrad(ctx, grad_input_nhwc,
                                                grad_input, kPermNhwcToNchw));
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

#define REGISTER_MUSA_MAXPOOL_GRAD(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("MaxPoolGrad").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMaxPoolGradOp<TYPE>)

REGISTER_MUSA_MAXPOOL_GRAD(float);
REGISTER_MUSA_MAXPOOL_GRAD(Eigen::half);
REGISTER_MUSA_MAXPOOL_GRAD(bfloat16);

#undef REGISTER_MUSA_MAXPOOL_GRAD

}  // namespace musa
}  // namespace tensorflow
