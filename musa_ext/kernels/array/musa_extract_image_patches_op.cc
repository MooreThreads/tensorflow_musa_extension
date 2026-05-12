#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <musa_runtime.h>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool FitsInInt64Mul(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return true;
  return a <= std::numeric_limits<int64_t>::max() / b;
}

inline bool FitsInInt64Add(int64_t a, int64_t b) {
  return a <= std::numeric_limits<int64_t>::max() - b;
}

Status ComputeOutputAndPadding2D(int64_t in_rows, int64_t in_cols,
                                 int64_t ksize_rows, int64_t ksize_cols,
                                 int stride_rows, int stride_cols,
                                 int rate_rows, int rate_cols,
                                 Padding padding, int64_t* out_rows,
                                 int64_t* out_cols, int* pad_top,
                                 int* pad_left) {
  const int64_t effective_rows = (ksize_rows - 1) * rate_rows + 1;
  const int64_t effective_cols = (ksize_cols - 1) * rate_cols + 1;

  if (padding == Padding::VALID) {
    *out_rows = in_rows < effective_rows
                    ? 0
                    : (in_rows - effective_rows) / stride_rows + 1;
    *out_cols = in_cols < effective_cols
                    ? 0
                    : (in_cols - effective_cols) / stride_cols + 1;
    *pad_top = 0;
    *pad_left = 0;
    return Status::OK();
  }

  if (padding == Padding::SAME) {
    *out_rows = in_rows == 0 ? 0 : (in_rows - 1) / stride_rows + 1;
    *out_cols = in_cols == 0 ? 0 : (in_cols - 1) / stride_cols + 1;

    int64_t pad_rows = 0;
    if (*out_rows > 0) {
      if (!FitsInInt64Mul(*out_rows - 1, stride_rows)) {
        return errors::InvalidArgument(
            "ExtractImagePatches row padding calculation overflow.");
      }
      const int64_t padded_rows_start = (*out_rows - 1) * stride_rows;
      if (!FitsInInt64Add(padded_rows_start, effective_rows)) {
        return errors::InvalidArgument(
            "ExtractImagePatches row padding calculation overflow.");
      }
      pad_rows =
          std::max<int64_t>(0, padded_rows_start + effective_rows - in_rows);
    }

    int64_t pad_cols = 0;
    if (*out_cols > 0) {
      if (!FitsInInt64Mul(*out_cols - 1, stride_cols)) {
        return errors::InvalidArgument(
            "ExtractImagePatches column padding calculation overflow.");
      }
      const int64_t padded_cols_start = (*out_cols - 1) * stride_cols;
      if (!FitsInInt64Add(padded_cols_start, effective_cols)) {
        return errors::InvalidArgument(
            "ExtractImagePatches column padding calculation overflow.");
      }
      pad_cols =
          std::max<int64_t>(0, padded_cols_start + effective_cols - in_cols);
    }

    *pad_top = static_cast<int>(pad_rows / 2);
    *pad_left = static_cast<int>(pad_cols / 2);
    return Status::OK();
  }

  return errors::InvalidArgument(
      "ExtractImagePatches only supports SAME/VALID padding.");
}

}  // namespace

template <typename T>
musaError_t LaunchExtractImagePatches(
    const T* input, T* output, int64_t total_elements, int64_t in_rows,
    int64_t in_cols, int64_t depth, int64_t out_rows, int64_t out_cols,
    int ksize_rows, int ksize_cols, int stride_rows, int stride_cols,
    int rate_rows, int rate_cols, int pad_top, int pad_left,
    musaStream_t stream);

template <typename T>
class MusaExtractImagePatchesOp : public MusaOpKernel {
 public:
  explicit MusaExtractImagePatchesOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksizes", &ksizes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rates", &rates_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_str_));

    OP_REQUIRES(
        ctx, ksizes_.size() == 4,
        errors::InvalidArgument(
            "ExtractImagePatches ksizes attr must have 4 elements."));
    OP_REQUIRES(
        ctx, strides_.size() == 4,
        errors::InvalidArgument(
            "ExtractImagePatches strides attr must have 4 elements."));
    OP_REQUIRES(
        ctx, rates_.size() == 4,
        errors::InvalidArgument(
            "ExtractImagePatches rates attr must have 4 elements."));

    OP_REQUIRES_OK(ctx, GetPaddingFromString(padding_str_, &padding_));
    OP_REQUIRES(ctx, padding_ == Padding::SAME || padding_ == Padding::VALID,
                errors::InvalidArgument(
                    "ExtractImagePatches only supports SAME/VALID padding."));

    OP_REQUIRES(ctx, ksizes_[0] == 1 && ksizes_[3] == 1,
                errors::InvalidArgument(
                    "ExtractImagePatches does not support patches across "
                    "batch/channel dimensions."));
    OP_REQUIRES(ctx, strides_[0] == 1 && strides_[3] == 1,
                errors::InvalidArgument(
                    "ExtractImagePatches does not support strides across "
                    "batch/channel dimensions."));
    OP_REQUIRES(ctx, rates_[0] == 1 && rates_[3] == 1,
                errors::InvalidArgument(
                    "ExtractImagePatches does not support rates across "
                    "batch/channel dimensions."));

    ksize_rows_ = ksizes_[1];
    ksize_cols_ = ksizes_[2];
    stride_rows_ = strides_[1];
    stride_cols_ = strides_[2];
    rate_rows_ = rates_[1];
    rate_cols_ = rates_[2];

    OP_REQUIRES(ctx, ksize_rows_ > 0 && ksize_cols_ > 0,
                errors::InvalidArgument(
                    "ExtractImagePatches spatial ksizes must be > 0."));
    OP_REQUIRES(ctx, stride_rows_ > 0 && stride_cols_ > 0,
                errors::InvalidArgument(
                    "ExtractImagePatches spatial strides must be > 0."));
    OP_REQUIRES(ctx, rate_rows_ > 0 && rate_cols_ > 0,
                errors::InvalidArgument(
                    "ExtractImagePatches spatial rates must be > 0."));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument(
                    "ExtractImagePatches input must be rank 4, got: ",
                    input.shape().DebugString()));

    const int64_t batch = input.dim_size(0);
    const int64_t in_rows = input.dim_size(1);
    const int64_t in_cols = input.dim_size(2);
    const int64_t depth = input.dim_size(3);

    int64_t out_rows = 0;
    int64_t out_cols = 0;
    int pad_top = 0;
    int pad_left = 0;
    OP_REQUIRES_OK(ctx, ComputeOutputAndPadding2D(
                            in_rows, in_cols, ksize_rows_, ksize_cols_,
                            stride_rows_, stride_cols_, rate_rows_, rate_cols_,
                            padding_, &out_rows, &out_cols, &pad_top,
                            &pad_left));

    OP_REQUIRES(
        ctx, FitsInInt64Mul(ksize_rows_, ksize_cols_) &&
                 FitsInInt64Mul(static_cast<int64_t>(ksize_rows_) *
                                    static_cast<int64_t>(ksize_cols_),
                                depth),
        errors::InvalidArgument("ExtractImagePatches output depth overflow."));
    const int64_t out_depth =
        static_cast<int64_t>(ksize_rows_) * ksize_cols_ * depth;

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(batch));
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(out_rows));
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(out_cols));
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(out_depth));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    const T* input_ptr = input.flat<T>().data();
    T* output_ptr = output->flat<T>().data();
    const int64_t total_elements = output->NumElements();
    auto stream = GetMusaStreamByCtx(ctx);

    musaError_t launch_status = LaunchExtractImagePatches<T>(
        input_ptr, output_ptr, total_elements, in_rows, in_cols, depth,
        out_rows, out_cols, ksize_rows_, ksize_cols_, stride_rows_, stride_cols_,
        rate_rows_, rate_cols_, pad_top, pad_left, stream);
    OP_REQUIRES(ctx, launch_status == musaSuccess,
                errors::Internal("ExtractImagePatches MUSA kernel launch "
                                 "failed: ",
                                 musaGetErrorString(launch_status)));
  }

 private:
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  std::vector<int32> rates_;
  std::string padding_str_;

  Padding padding_ = Padding::SAME;
  int ksize_rows_ = 1;
  int ksize_cols_ = 1;
  int stride_rows_ = 1;
  int stride_cols_ = 1;
  int rate_rows_ = 1;
  int rate_cols_ = 1;
};

#define REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(TYPE)                 \
  REGISTER_KERNEL_BUILDER(Name("ExtractImagePatches")             \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<TYPE>("T"),         \
                          MusaExtractImagePatchesOp<TYPE>)

REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(float);
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(double);
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(int32);
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(int64);
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(Eigen::half);
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES(bfloat16);

#undef REGISTER_MUSA_EXTRACT_IMAGE_PATCHES

}  // namespace musa
}  // namespace tensorflow
