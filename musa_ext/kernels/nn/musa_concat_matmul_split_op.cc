#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

std::vector<int64_t> MakeContiguousStrides(const TensorShape& shape) {
  std::vector<int64_t> strides(shape.dims(), 1);
  for (int i = shape.dims() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }
  return strides;
}

TensorShape MakeSlicedShape(const TensorShape& shape, int slice_dim,
                            int64_t slice_size) {
  TensorShape sliced = shape;
  sliced.set_dim(slice_dim, slice_size);
  return sliced;
}

Status CreateSliceView(const Tensor& tensor, int slice_dim, int64_t slice_begin,
                       int64_t slice_size, mFormat format, mTensor* view) {
  if (tensor.dims() <= 0) {
    return errors::InvalidArgument(
        "CreateSliceView requires tensor rank >= 1, got ", tensor.dims());
  }
  if (slice_dim < 0 || slice_dim >= tensor.dims()) {
    return errors::InvalidArgument("slice_dim out of range: ", slice_dim,
                                   " for rank ", tensor.dims());
  }
  if (slice_begin < 0 || slice_size < 0 ||
      slice_begin + slice_size > tensor.dim_size(slice_dim)) {
    return errors::InvalidArgument(
        "Invalid slice range [", slice_begin, ", ", slice_begin + slice_size,
        ") for dim ", slice_dim, " with size ",
        tensor.dim_size(slice_dim));
  }

  std::vector<int64_t> dims(tensor.dims());
  for (int i = 0; i < tensor.dims(); ++i) {
    dims[i] = tensor.dim_size(i);
  }
  std::vector<int64_t> strides = MakeContiguousStrides(tensor.shape());
  const int64_t element_offset = slice_begin * strides[slice_dim];
  dims[slice_dim] = slice_size;

  *view = CreateMTensor(tensor, format);
  auto status = view->SetAddr(
      const_cast<char*>(tensor.tensor_data().data()) +
      element_offset * DataTypeSize(tensor.dtype()));
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("SetAddr failed while building slice view. "
                            "Status: ",
                            static_cast<int>(status));
  }

  status = view->SetNdInfo(static_cast<int>(dims.size()), dims.data(),
                           strides.data());
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("SetNdInfo failed while building slice view. "
                            "Status: ",
                            static_cast<int>(status));
  }
  return Status::OK();
}

Status ConfigureBatchMatMulTensor(mTensor* tensor, const TensorShape& shape,
                                  int64_t out_batch) {
  const int dims = shape.dims();
  if (dims < 2) {
    return errors::InvalidArgument("BatchMatMul tensor rank must be >= 2, got ",
                                   dims);
  }

  const int64_t rows = shape.dim_size(dims - 2);
  const int64_t cols = shape.dim_size(dims - 1);
  const int64_t batch = shape.num_elements() / (rows * cols);

  ::musa::dnn::Status status = ::musa::dnn::Status::SUCCESS;
  if (dims != 3) {
    if (batch == 1 && out_batch > 1) {
      status = tensor->SetNdInfo({out_batch, rows, cols}, {0, cols, 1});
    } else {
      status = tensor->SetNdInfo({batch, rows, cols}, {rows * cols, cols, 1});
    }
  } else if (batch == 1 && out_batch > 1) {
    status = tensor->SetNdInfo({out_batch, rows, cols}, {0, cols, 1});
  }

  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(
        "Failed to configure BatchMatMul tensor view. Status: ",
        static_cast<int>(status));
  }
  return Status::OK();
}

}  // namespace

template <typename T>
class MusaConcatMatMulSplitOp : public MusaOpKernel {
 public:
  explicit MusaConcatMatMulSplitOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_concat", &num_concat_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_input_idx", &concat_input_idx_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_args", &num_args_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_alpha", &activation_alpha_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_outputs", &num_outputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_axis", &slice_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_sizes", &slice_sizes_));

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));
    if (fused_ops.size() == 1 && fused_ops[0] == "BiasAdd" && num_args_ == 1) {
      fusion_type_ = FusionType::BIAS_ADD;
    } else if (fused_ops.size() == 2 && fused_ops[0] == "BiasAdd" &&
               fused_ops[1] == "Relu" && num_args_ == 1) {
      fusion_type_ = FusionType::BIAS_ADD_RELU;
    } else if (fused_ops.size() == 2 && fused_ops[0] == "BiasAdd" &&
               fused_ops[1] == "LeakyRelu" && num_args_ == 1) {
      fusion_type_ = FusionType::BIAS_ADD_LEAKY_RELU;
    } else {
      fusion_type_ = FusionType::NONE;
    }

    OP_REQUIRES(
        ctx, num_outputs_ >= 1,
        errors::InvalidArgument("num_outputs must be >= 1, got ",
                                num_outputs_));
    OP_REQUIRES(
        ctx, static_cast<int>(slice_sizes_.size()) == num_outputs_,
        errors::InvalidArgument("slice_sizes size must equal num_outputs, got ",
                                slice_sizes_.size(), " vs ", num_outputs_));
    for (int i = 0; i < num_outputs_; ++i) {
      OP_REQUIRES(ctx, slice_sizes_[i] > 0,
                  errors::InvalidArgument("slice_sizes must be > 0, got ",
                                          slice_sizes_[i], " at index ", i));
    }

    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return false;
    }();
    tf32_enabled_ = tf32_enabled_global;
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);
    OP_REQUIRES(ctx, ctx->num_outputs() == num_outputs_,
                errors::InvalidArgument("Expected ", num_outputs_,
                                        " outputs, got ",
                                        ctx->num_outputs()));

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    const Tensor& axis_tensor = ctx->input(num_concat_);
    const int64 axis_val = axis_tensor.scalar<int32>()();

    int first_non_empty_idx = -1;
    std::vector<const Tensor*> concat_inputs;
    concat_inputs.reserve(num_concat_);
    int64_t concat_dim_total = 0;
    for (int i = 0; i < num_concat_; ++i) {
      const Tensor& t = ctx->input(i);
      concat_inputs.push_back(&t);
      if (t.NumElements() > 0 && first_non_empty_idx < 0) {
        first_non_empty_idx = i;
      }
    }

    const Tensor& ref =
        ctx->input(first_non_empty_idx == -1 ? 0 : first_non_empty_idx);
    const int normalized_concat_axis =
        axis_val < 0 ? axis_val + ref.dims() : axis_val;
    OP_REQUIRES(ctx,
                normalized_concat_axis >= 0 &&
                    normalized_concat_axis < ref.dims(),
                errors::InvalidArgument("Concat axis out of range: ", axis_val,
                                        " for rank ", ref.dims()));

    for (int i = 0; i < num_concat_; ++i) {
      concat_dim_total += ctx->input(i).dim_size(normalized_concat_axis);
    }

    TensorShape concat_shape = ref.shape();
    concat_shape.set_dim(normalized_concat_axis, concat_dim_total);

    Tensor concat_out_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ref.dtype(), concat_shape, &concat_out_tensor));

    std::vector<::musa::dnn::Tensor> mudnn_ins;
    mudnn_ins.reserve(num_concat_);
    for (int i = 0; i < num_concat_; ++i) {
      if (ctx->input(i).NumElements() > 0) {
        mudnn_ins.push_back(CreateMTensor(ctx->input(i), format_));
      }
    }

    if (!mudnn_ins.empty() && concat_out_tensor.NumElements() > 0) {
      ::musa::dnn::Tensor mudnn_concat_out =
          CreateMTensor(concat_out_tensor, format_);
      ::musa::dnn::Concat concat_op;
      concat_op.SetAxis(normalized_concat_axis);
      auto status = concat_op.Run(handle, mudnn_concat_out, mudnn_ins.size(),
                                  mudnn_ins.data());
      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal("MUSA Concat failed in MusaConcatMatMulSplit. "
                           "Status: ",
                           static_cast<int>(status)));
    }

    const Tensor& other_input = ctx->input(num_concat_ + 1);
    const Tensor& in0 =
        (concat_input_idx_ == 0) ? concat_out_tensor : other_input;
    const Tensor& in1 =
        (concat_input_idx_ == 1) ? concat_out_tensor : other_input;

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for MatMul in MusaConcatMatMulSplit"));
    OP_REQUIRES(ctx, in0.dims() >= 2 && in1.dims() >= 2,
                errors::InvalidArgument(
                    "Input tensors must have rank >= 2 in MusaConcatMatMulSplit"));

    int64 m = trans_a_ ? in0.dim_size(in0.dims() - 1)
                       : in0.dim_size(in0.dims() - 2);
    int64 n = trans_b_ ? in1.dim_size(in1.dims() - 2)
                       : in1.dim_size(in1.dims() - 1);

    const Tensor* bias = nullptr;
    if (HasBiasFusion()) {
      OP_REQUIRES(ctx, num_args_ == 1,
                  errors::InvalidArgument(
                      "MusaConcatMatMulSplit BiasAdd expects exactly 1 "
                      "fused arg"));
      OP_REQUIRES(ctx, ctx->num_inputs() >= num_concat_ + 3,
                  errors::InvalidArgument(
                      "MusaConcatMatMulSplit BiasAdd missing fused bias "
                      "input"));
      bias = &ctx->input(num_concat_ + 2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias->shape()),
                  errors::InvalidArgument("Bias must be 1-D, got shape ",
                                          bias->shape().DebugString()));
      OP_REQUIRES(ctx, bias->dim_size(0) == n,
                  errors::InvalidArgument(
                      "Bias size mismatch in MusaConcatMatMulSplit: expected ",
                      n, ", got ", bias->dim_size(0)));
    }

    TensorShape mm_out_shape = bcast.output_batch_shape();
    mm_out_shape.AddDim(m);
    mm_out_shape.AddDim(n);

    const int normalized_slice_axis =
        slice_axis_ < 0 ? slice_axis_ + mm_out_shape.dims() : slice_axis_;
    OP_REQUIRES(ctx,
                normalized_slice_axis >= 0 &&
                    normalized_slice_axis < mm_out_shape.dims(),
                errors::InvalidArgument("slice_axis out of range: ", slice_axis_,
                                        " for output rank ",
                                        mm_out_shape.dims()));

    int64_t consumed_axis = 0;
    for (int size : slice_sizes_) {
      consumed_axis += size;
    }
    OP_REQUIRES(
        ctx, consumed_axis <= mm_out_shape.dim_size(normalized_slice_axis),
        errors::InvalidArgument(
            "slice_sizes exceed output dimension: consumed=", consumed_axis,
            ", available=", mm_out_shape.dim_size(normalized_slice_axis)));

    if (CanUseDirectOutputColumnSplitFastPath(mm_out_shape,
                                              normalized_slice_axis)) {
      OP_REQUIRES_OK(ctx, RunDirectOutputColumnSplitFastPath(
                              ctx, in0, in1, bias, mm_out_shape,
                              normalized_slice_axis));
      return;
    }

    if (CanUseDirectOutputRowSplitFastPath(mm_out_shape,
                                           normalized_slice_axis)) {
      OP_REQUIRES_OK(ctx, RunDirectOutputRowSplitFastPath(
                              ctx, in0, in1, bias, mm_out_shape,
                              normalized_slice_axis));
      return;
    }

    Tensor matmul_out_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ref.dtype(), mm_out_shape, &matmul_out_tensor));

    if (HasBiasFusion() && in0.dims() == 2 && in1.dims() == 2) {
      OP_REQUIRES_OK(ctx, Run2DMatMulWithBias(ctx, in0, in1, *bias,
                                              &matmul_out_tensor));
    } else {
      OP_REQUIRES_OK(ctx, RunMatMul(ctx, in0, in1, &matmul_out_tensor));
      if (HasBiasFusion()) {
        OP_REQUIRES_OK(ctx, ApplyBiasAddInPlace(ctx, *bias, mm_out_shape,
                                                &matmul_out_tensor));
      }
    }

    if (HasActivationFusion()) {
      OP_REQUIRES_OK(ctx, ApplyActivationInPlace(ctx, &matmul_out_tensor));
    }

    ::musa::dnn::Tensor mt_input = CreateMTensor(matmul_out_tensor, format_);
    std::vector<int64_t> starts(mm_out_shape.dims(), 0);
    int64_t current_offset = 0;

    for (int i = 0; i < num_outputs_; ++i) {
      TensorShape output_shape = mm_out_shape;
      output_shape.set_dim(normalized_slice_axis, slice_sizes_[i]);

      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &output));
      if (output->NumElements() == 0) {
        current_offset += slice_sizes_[i];
        continue;
      }

      starts[normalized_slice_axis] = current_offset;
      current_offset += slice_sizes_[i];

      ::musa::dnn::Tensor mt_output = CreateMTensor(*output, format_);
      ::musa::dnn::Permute slice_op;
      auto status =
          slice_op.ConfigDimStrideForSlice(mt_output, mt_input, starts.data());
      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal(
              "MUSA ConfigDimStrideForSlice failed in MusaConcatMatMulSplit. "
              "Status: ",
              static_cast<int>(status)));

      status = slice_op.Run(handle, mt_output, mt_input);
      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal("MUSA Slice run failed in MusaConcatMatMulSplit. "
                           "Status: ",
                           static_cast<int>(status)));
    }
  }

 private:
  bool HasBiasFusion() const { return fusion_type_ != FusionType::NONE; }

  bool HasActivationFusion() const {
    return fusion_type_ == FusionType::BIAS_ADD_RELU ||
           fusion_type_ == FusionType::BIAS_ADD_LEAKY_RELU;
  }

  bool CanUseDirectOutputColumnSplitFastPath(
      const TensorShape& mm_out_shape, int normalized_slice_axis) const {
    return mm_out_shape.dims() >= 2 &&
           normalized_slice_axis == mm_out_shape.dims() - 1;
  }

  Status RunDirectOutputColumnSplitFastPath(OpKernelContext* ctx,
                                            const Tensor& in0,
                                            const Tensor& in1,
                                            const Tensor* bias,
                                            const TensorShape& mm_out_shape,
                                            int normalized_slice_axis) {
    const int slice_dim_in_b = trans_b_ ? in1.dims() - 2 : in1.dims() - 1;
    int64_t current_offset = 0;

    for (int i = 0; i < num_outputs_; ++i) {
      const TensorShape rhs_slice_shape =
          MakeSlicedShape(in1.shape(), slice_dim_in_b, slice_sizes_[i]);
      const TensorShape output_shape =
          MakeSlicedShape(mm_out_shape, normalized_slice_axis, slice_sizes_[i]);

      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, output_shape, &output));
      if (output->NumElements() == 0) {
        current_offset += slice_sizes_[i];
        continue;
      }

      mTensor mt_a = CreateMTensor(in0, format_);
      mTensor mt_b;
      TF_RETURN_IF_ERROR(CreateSliceView(in1, slice_dim_in_b, current_offset,
                                         slice_sizes_[i], format_, &mt_b));

      if (HasBiasFusion() && output_shape.dims() == 2) {
        mTensor mt_bias;
        TF_RETURN_IF_ERROR(
            CreateSliceView(*bias, 0, current_offset, slice_sizes_[i], format_,
                            &mt_bias));
        TF_RETURN_IF_ERROR(
            Run2DMatMulWithBias(ctx, mt_a, mt_b, mt_bias, output));
      } else {
        TF_RETURN_IF_ERROR(RunMatMulWithViews(
            ctx, in0.shape(), mt_a, rhs_slice_shape, mt_b, output_shape,
            output));
        if (HasBiasFusion()) {
          mTensor mt_bias;
          TF_RETURN_IF_ERROR(
              CreateSliceView(*bias, 0, current_offset, slice_sizes_[i],
                              format_, &mt_bias));
          TF_RETURN_IF_ERROR(ApplyBiasAddInPlace(
              ctx, mt_bias, slice_sizes_[i], output_shape, output));
        }
      }

      if (HasActivationFusion()) {
        TF_RETURN_IF_ERROR(ApplyActivationInPlace(ctx, output));
      }
      current_offset += slice_sizes_[i];
    }

    return Status::OK();
  }

  bool CanUseDirectOutputRowSplitFastPath(const TensorShape& mm_out_shape,
                                          int normalized_slice_axis) const {
    return mm_out_shape.dims() >= 2 &&
           normalized_slice_axis == mm_out_shape.dims() - 2;
  }

  Status RunDirectOutputRowSplitFastPath(OpKernelContext* ctx, const Tensor& in0,
                                         const Tensor& in1, const Tensor* bias,
                                         const TensorShape& mm_out_shape,
                                         int normalized_slice_axis) {
    const int slice_dim_in_a = trans_a_ ? in0.dims() - 1 : in0.dims() - 2;
    int64_t current_offset = 0;

    for (int i = 0; i < num_outputs_; ++i) {
      const TensorShape lhs_slice_shape =
          MakeSlicedShape(in0.shape(), slice_dim_in_a, slice_sizes_[i]);
      const TensorShape output_shape =
          MakeSlicedShape(mm_out_shape, normalized_slice_axis, slice_sizes_[i]);

      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, output_shape, &output));
      if (output->NumElements() == 0) {
        current_offset += slice_sizes_[i];
        continue;
      }

      mTensor mt_a;
      TF_RETURN_IF_ERROR(CreateSliceView(in0, slice_dim_in_a, current_offset,
                                         slice_sizes_[i], format_, &mt_a));
      mTensor mt_b = CreateMTensor(in1, format_);

      if (HasBiasFusion() && output_shape.dims() == 2) {
        mTensor mt_bias = CreateMTensor(*bias, format_);
        TF_RETURN_IF_ERROR(
            Run2DMatMulWithBias(ctx, mt_a, mt_b, mt_bias, output));
      } else {
        TF_RETURN_IF_ERROR(RunMatMulWithViews(
            ctx, lhs_slice_shape, mt_a, in1.shape(), mt_b, output_shape,
            output));
        if (HasBiasFusion()) {
          TF_RETURN_IF_ERROR(
              ApplyBiasAddInPlace(ctx, *bias, output_shape, output));
        }
      }

      if (HasActivationFusion()) {
        TF_RETURN_IF_ERROR(ApplyActivationInPlace(ctx, output));
      }
      current_offset += slice_sizes_[i];
    }
    return Status::OK();
  }

  Status Run2DMatMul(OpKernelContext* ctx, const mTensor& mt_a,
                     const mTensor& mt_b, Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output, format_);

    mMatMul mm_op;
    mm_op.SetTranspose(trans_a_, trans_b_);
    mm_op.SetAlpha(1.0);
    mm_op.SetBeta(0.0);
    auto status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA MatMul failed in MusaConcatMatMulSplit. "
                              "Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status RunMatMulWithViews(OpKernelContext* ctx, const TensorShape& lhs_shape,
                            mTensor mt_a, const TensorShape& rhs_shape,
                            mTensor mt_b, const TensorShape& output_shape,
                            Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output, format_);

    ::musa::dnn::Status status;
    if (lhs_shape.dims() == 2 && rhs_shape.dims() == 2) {
      mMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    } else {
      const int64_t m = output_shape.dim_size(output_shape.dims() - 2);
      const int64_t n = output_shape.dim_size(output_shape.dims() - 1);
      const int64_t out_batch = output_shape.num_elements() / (m * n);

      TF_RETURN_IF_ERROR(
          ConfigureBatchMatMulTensor(&mt_a, lhs_shape, out_batch));
      TF_RETURN_IF_ERROR(
          ConfigureBatchMatMulTensor(&mt_b, rhs_shape, out_batch));
      if (output_shape.dims() > 3) {
        status = mt_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});
        if (status != ::musa::dnn::Status::SUCCESS) {
          return errors::Internal(
              "Failed to configure BatchMatMul output view. Status: ",
              static_cast<int>(status));
        }
      } else if (output_shape.dims() == 2) {
        status = mt_out.SetNdInfo({1, m, n}, {m * n, n, 1});
        if (status != ::musa::dnn::Status::SUCCESS) {
          return errors::Internal(
              "Failed to configure BatchMatMul 2D output view. Status: ",
              static_cast<int>(status));
        }
      }

      mBatchMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    }

    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA MatMul failed in MusaConcatMatMulSplit. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  Status RunMatMul(OpKernelContext* ctx, const Tensor& in0, const Tensor& in1,
                   Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_a = CreateMTensor(in0, format_);
    mTensor mt_b = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    ::musa::dnn::Status status;
    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    } else {
      mBatchMatMul mm_op;
      mm_op.SetTranspose(trans_a_, trans_b_);
      mm_op.SetAlpha(1.0);
      mm_op.SetBeta(0.0);
      status = mm_op.Run(handle, mt_out, mt_a, mt_b);
    }

    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA MatMul failed in MusaConcatMatMulSplit. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  Status Run2DMatMulWithBias(OpKernelContext* ctx, const Tensor& a,
                             const Tensor& b, const Tensor& bias,
                             Tensor* output) {
    mTensor mt_a = CreateMTensor(a, format_);
    mTensor mt_b = CreateMTensor(b, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    return Run2DMatMulWithBias(ctx, mt_a, mt_b, mt_bias, output);
  }

  Status Run2DMatMulWithBias(OpKernelContext* ctx, const mTensor& mt_a,
                             const mTensor& mt_b, const mTensor& mt_bias,
                             Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output, format_);

    mMatMul op;
    op.SetTranspose(trans_a_, trans_b_);
    op.SetAlpha(1.0);
    op.SetBeta(0.0);

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      auto deleter = [tf_allocator](void* p) {
        if (p) {
          tf_allocator->DeallocateRaw(p);
        }
      };
      return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };
    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA MatMul+BiasAdd epilogue failed in "
                              "MusaConcatMatMulSplit. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyBiasAddInPlace(OpKernelContext* ctx, const Tensor& bias,
                             const TensorShape& out_shape, Tensor* output) {
    mTensor mt_bias = CreateMTensor(bias, format_);
    return ApplyBiasAddInPlace(ctx, mt_bias, bias.dim_size(0), out_shape,
                               output);
  }

  Status ApplyBiasAddInPlace(OpKernelContext* ctx, mTensor mt_bias,
                             int64_t bias_size, const TensorShape& out_shape,
                             Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output, format_);

    const int dims_cnt = out_shape.dims();
    const int channel_dim = dims_cnt - 1;
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = bias_size;
    b_strides[channel_dim] = 1;
    auto status = mt_bias.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "Failed to configure BiasAdd view in MusaConcatMatMulSplit. Status: ",
          static_cast<int>(status));
    }

    mBinary bias_add_op;
    bias_add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    status = bias_add_op.Run(handle, mt_out, mt_out, mt_bias);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA BiasAdd failed in MusaConcatMatMulSplit. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  Status ApplyActivationInPlace(OpKernelContext* ctx, Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_out = CreateMTensor(*output, format_);
    mUnary activation_op;

    if (fusion_type_ == FusionType::BIAS_ADD_RELU) {
      activation_op.SetMode(::musa::dnn::Unary::Mode::RELU);
    } else if (fusion_type_ == FusionType::BIAS_ADD_LEAKY_RELU) {
      activation_op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU);
      activation_op.SetAlpha(static_cast<double>(activation_alpha_));
    } else {
      return Status::OK();
    }

    auto status = activation_op.Run(handle, mt_out, mt_out);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "MUSA activation failed in MusaConcatMatMulSplit. Status: ",
          static_cast<int>(status));
    }
    return Status::OK();
  }

  enum class FusionType {
    NONE,
    BIAS_ADD,
    BIAS_ADD_RELU,
    BIAS_ADD_LEAKY_RELU,
  };

  bool trans_a_ = false;
  bool trans_b_ = false;
  int num_concat_ = 0;
  int concat_input_idx_ = 0;
  int num_args_ = 0;
  float activation_alpha_ = 0.2f;
  bool tf32_enabled_ = false;
  int num_outputs_ = 0;
  int slice_axis_ = 0;
  std::vector<int> slice_sizes_;
  FusionType fusion_type_ = FusionType::NONE;
};

#define REGISTER_MUSA_CONCAT_MATMUL_SPLIT(TYPE)                \
  REGISTER_KERNEL_BUILDER(Name("MusaConcatMatMulSplit")        \
                              .Device("MUSA")                  \
                              .TypeConstraint<TYPE>("T")       \
                              .HostMemory("axis"),             \
                          MusaConcatMatMulSplitOp<TYPE>);

REGISTER_MUSA_CONCAT_MATMUL_SPLIT(float);
REGISTER_MUSA_CONCAT_MATMUL_SPLIT(Eigen::half);
REGISTER_MUSA_CONCAT_MATMUL_SPLIT(double);
REGISTER_MUSA_CONCAT_MATMUL_SPLIT(bfloat16);

#undef REGISTER_MUSA_CONCAT_MATMUL_SPLIT

REGISTER_OP("MusaConcatMatMulSplit")
    .Input("inputs: num_concat * T")
    .Input("axis: int32")
    .Input("other: T")
    .Input("args: num_args * T")
    .Output("outputs: num_outputs * T")
    .Attr("T: {float, half, bfloat16, double}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("num_concat: int >= 1")
    .Attr("concat_input_idx: int")
    .Attr("fused_ops: list(string) = []")
    .Attr("num_args: int >= 0 = 0")
    .Attr("activation_alpha: float = 0.2")
    .Attr("num_outputs: int >= 1")
    .Attr("slice_axis: int")
    .Attr("slice_sizes: list(int)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_outputs = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("num_outputs", &num_outputs));
      for (int i = 0; i < num_outputs; ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return Status::OK();
    });

}  // namespace musa
}  // namespace tensorflow
