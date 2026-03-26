#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "../utils_op.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

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

static const std::vector<int64_t> kPermNchwToNhwc = {0, 2, 3, 1};
static const std::vector<int64_t> kPermNhwcToNchw = {0, 3, 1, 2};

}  // namespace

template <typename T>
class MusaFusedBatchNormOp : public MusaOpKernel {
 public:
  explicit MusaFusedBatchNormOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("exponential_avg_factor", &exp_avg_factor_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    is_nhwc_ = (data_format_str == "NHWC");
  }

  // BatchNorm is computationally intensive (reduction operations)
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "FusedBatchNorm");

    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);
    const Tensor& est_mean = ctx->input(3);
    const Tensor& est_var = ctx->input(4);

    Tensor* y = nullptr;
    MUSA_KERNEL_TRACE_START("Mem Alloc");
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    Tensor* batch_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, scale.shape(), &batch_mean));
    Tensor* batch_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, scale.shape(), &batch_var));
    Tensor* saved_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, scale.shape(), &saved_mean));
    Tensor* saved_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, scale.shape(), &saved_var));
    Tensor* reserve_3 = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(5, TensorShape({}), &reserve_3));
    MUSA_KERNEL_TRACE_END("Mem Alloc");

    auto* device = GetDeviceByCtx(ctx);
    auto& handle = device->mudnn_handle();
    auto stream = device->GetStream();
    handle.SetAllowTF32(false);

    std::vector<Tensor> workspace_holder;
    auto internal_maintainer = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      Tensor temp;
      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      workspace_holder.push_back(temp);
      return ::musa::dnn::MemoryHandler(temp.flat<uint8_t>().data(),
                                        [](void*) {});
    };
    auto maintainer = device->GetMemMaintainer(internal_maintainer);

    const Tensor* x_ptr = &x;
    Tensor x_nhwc;
    Tensor y_nhwc;
    if (!is_nhwc_) {
      OP_REQUIRES(ctx, x.dims() == 4,
                  errors::InvalidArgument(
                      "FusedBatchNorm NCHW fallback expects rank-4 input, got ",
                      x.dims()));
      TensorShape nhwc_shape(
          {x.dim_size(0), x.dim_size(2), x.dim_size(3), x.dim_size(1)});
      MUSA_KERNEL_TRACE_START("Permute Alloc");
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), nhwc_shape, &x_nhwc));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(y->dtype(), nhwc_shape, &y_nhwc));
      MUSA_KERNEL_TRACE_END("Permute Alloc");
      MUSA_KERNEL_TRACE_START("Permute In");
      OP_REQUIRES_OK(ctx,
                     PermuteTensorOnMusa(ctx, x, &x_nhwc, kPermNchwToNhwc));
      MUSA_KERNEL_TRACE_END("Permute In");
      x_ptr = &x_nhwc;
    }

    // Native NCHW BatchNorm is numerically unstable on current muDNN, so the
    // fallback path permutes tensors to NHWC and reuses the validated NHWC
    // kernel implementation.
    mFormat data_fmt = mFormat::NHWC;

    MUSA_KERNEL_TRACE_START("Tensor Wrap");
    mTensor mt_x = CreateMTensor(*x_ptr, data_fmt);
    mTensor mt_y = CreateMTensor(is_nhwc_ ? *y : y_nhwc, data_fmt);

    mTensor mt_scale = CreateMTensor(scale, mFormat::NCHW);
    mTensor mt_offset = CreateMTensor(offset, mFormat::NCHW);
    mTensor mt_fresh_mean = CreateMTensor(*saved_mean, mFormat::NCHW);
    mTensor mt_fresh_var = CreateMTensor(*saved_var, mFormat::NCHW);
    MUSA_KERNEL_TRACE_END("Tensor Wrap");

    mBatchNorm bn_op;
    MUSA_KERNEL_TRACE_START("Set Attr");
    bn_op.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL);
    bn_op.SetEpsilon(epsilon_);
    bn_op.SetTraining(is_training_);
    MUSA_KERNEL_TRACE_END("Set Attr");

    mStatus status;
    if (is_training_) {
      Tensor temp_acc_mean, temp_acc_var;
      MUSA_KERNEL_TRACE_START("Workspace Alloc");
      ctx->allocate_temp(DT_FLOAT, scale.shape(), &temp_acc_mean);
      ctx->allocate_temp(DT_FLOAT, scale.shape(), &temp_acc_var);

      musaMemsetAsync(temp_acc_mean.flat<float>().data(), 0,
                      temp_acc_mean.NumElements() * sizeof(float), stream);
      musaMemsetAsync(temp_acc_var.flat<float>().data(), 0,
                      temp_acc_var.NumElements() * sizeof(float), stream);
      MUSA_KERNEL_TRACE_END("Workspace Alloc");

      MUSA_KERNEL_TRACE_START("Tensor Wrap");
      mTensor mt_acc_mean = CreateMTensor(temp_acc_mean, mFormat::NCHW);
      mTensor mt_acc_var = CreateMTensor(temp_acc_var, mFormat::NCHW);
      MUSA_KERNEL_TRACE_END("Tensor Wrap");

      MUSA_KERNEL_TRACE_START("Kernel");
      status =
          bn_op.RunComposite(handle, mt_y, mt_x, mt_acc_mean, mt_acc_var,
                             mt_fresh_mean, mt_fresh_var, mt_scale, mt_offset,
                             (double)exp_avg_factor_, maintainer);
      MUSA_KERNEL_TRACE_END("Kernel");

      if (status == mStatus::SUCCESS) {
        size_t copy_size = saved_mean->NumElements() * sizeof(float);
        MUSA_KERNEL_TRACE_START("Copy Stats");
        musaMemcpyAsync(batch_mean->flat<float>().data(),
                        saved_mean->flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);
        musaMemcpyAsync(batch_var->flat<float>().data(),
                        saved_var->flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);
        MUSA_KERNEL_TRACE_END("Copy Stats");
      }

    } else {
      MUSA_KERNEL_TRACE_START("Tensor Wrap");
      mTensor mt_est_mean = CreateMTensor(est_mean, mFormat::NCHW);
      mTensor mt_est_var = CreateMTensor(est_var, mFormat::NCHW);
      MUSA_KERNEL_TRACE_END("Tensor Wrap");
      MUSA_KERNEL_TRACE_START("Kernel");
      status = bn_op.RunPure(handle, mt_y, mt_x, mt_est_mean, mt_est_var,
                             mt_scale, mt_offset);
      MUSA_KERNEL_TRACE_END("Kernel");
    }

    if (status == mStatus::SUCCESS && !is_nhwc_) {
      MUSA_KERNEL_TRACE_START("Permute Out");
      OP_REQUIRES_OK(ctx,
                     PermuteTensorOnMusa(ctx, y_nhwc, y, kPermNhwcToNchw));
      MUSA_KERNEL_TRACE_END("Permute Out");
    }

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA BN Forward failed."));
  }

 private:
  float epsilon_;
  bool is_training_;
  float exp_avg_factor_;
  bool is_nhwc_;
};

template <typename T>
class MusaFusedBatchNormGradOp : public MusaOpKernel {
 public:
  explicit MusaFusedBatchNormGradOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    is_nhwc_ = (data_format_str == "NHWC");
  }

  // BatchNormGrad is computationally intensive
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "FusedBatchNormGrad");

    const Tensor& dy = ctx->input(0);
    const Tensor& x = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& saved_mean = ctx->input(3);
    const Tensor& saved_var = ctx->input(4);

    Tensor* dx = nullptr;
    MUSA_KERNEL_TRACE_START("Mem Alloc");
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    Tensor* d_scale = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, scale.shape(), &d_scale));
    Tensor* d_offset = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, scale.shape(), &d_offset));

    Tensor* d_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, scale.shape(), &d_mean));
    Tensor* d_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, scale.shape(), &d_var));
    MUSA_KERNEL_TRACE_END("Mem Alloc");

    auto* device = GetDeviceByCtx(ctx);
    auto& handle = device->mudnn_handle();
    auto stream = device->GetStream();
    handle.SetAllowTF32(false);

    musaMemsetAsync(d_scale->flat<float>().data(), 0,
                    d_scale->NumElements() * sizeof(float), stream);
    musaMemsetAsync(d_offset->flat<float>().data(), 0,
                    d_offset->NumElements() * sizeof(float), stream);
    musaMemsetAsync(d_mean->flat<float>().data(), 0,
                    d_mean->NumElements() * sizeof(float), stream);
    musaMemsetAsync(d_var->flat<float>().data(), 0,
                    d_var->NumElements() * sizeof(float), stream);

    std::vector<Tensor> workspace_holder;
    auto maintainer_func = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      Tensor temp;
      ctx->allocate_temp(DT_UINT8, TensorShape({static_cast<int64_t>(size)}),
                         &temp);
      workspace_holder.push_back(temp);
      return ::musa::dnn::MemoryHandler(temp.flat<uint8_t>().data(),
                                        [](void*) {});
    };
    auto maintainer = device->GetMemMaintainer(maintainer_func);

    const Tensor* x_ptr = &x;
    const Tensor* dy_ptr = &dy;
    Tensor x_nhwc;
    Tensor dy_nhwc;
    Tensor dx_nhwc;
    if (!is_nhwc_) {
      OP_REQUIRES(ctx, x.dims() == 4,
                  errors::InvalidArgument(
                      "FusedBatchNormGrad NCHW fallback expects rank-4 input, got ",
                      x.dims()));
      TensorShape nhwc_shape(
          {x.dim_size(0), x.dim_size(2), x.dim_size(3), x.dim_size(1)});
      MUSA_KERNEL_TRACE_START("Permute Alloc");
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), nhwc_shape, &x_nhwc));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(dy.dtype(), nhwc_shape, &dy_nhwc));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(dx->dtype(), nhwc_shape, &dx_nhwc));
      MUSA_KERNEL_TRACE_END("Permute Alloc");
      MUSA_KERNEL_TRACE_START("Permute In");
      OP_REQUIRES_OK(ctx,
                     PermuteTensorOnMusa(ctx, x, &x_nhwc, kPermNchwToNhwc));
      OP_REQUIRES_OK(ctx,
                     PermuteTensorOnMusa(ctx, dy, &dy_nhwc, kPermNchwToNhwc));
      MUSA_KERNEL_TRACE_END("Permute In");
      x_ptr = &x_nhwc;
      dy_ptr = &dy_nhwc;
    }

    // Keep backward consistent with the forward fallback by running BatchNorm
    // in NHWC and permuting gradients back to NCHW when needed.
    mFormat data_fmt = mFormat::NHWC;

    MUSA_KERNEL_TRACE_START("Tensor Wrap");
    mTensor mt_dy = CreateMTensor(*dy_ptr, data_fmt);
    mTensor mt_x = CreateMTensor(*x_ptr, data_fmt);
    mTensor mt_dx = CreateMTensor(is_nhwc_ ? *dx : dx_nhwc, data_fmt);

    mTensor mt_scale = CreateMTensor(scale, mFormat::NCHW);
    mTensor mt_saved_mean = CreateMTensor(saved_mean, mFormat::NCHW);
    mTensor mt_saved_var = CreateMTensor(saved_var, mFormat::NCHW);

    mTensor mt_d_scale = CreateMTensor(*d_scale, mFormat::NCHW);
    mTensor mt_d_offset = CreateMTensor(*d_offset, mFormat::NCHW);
    mTensor mt_d_mean = CreateMTensor(*d_mean, mFormat::NCHW);
    mTensor mt_d_var = CreateMTensor(*d_var, mFormat::NCHW);
    MUSA_KERNEL_TRACE_END("Tensor Wrap");

    mBatchNorm bn_op;
    MUSA_KERNEL_TRACE_START("Set Attr");
    bn_op.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL);
    bn_op.SetEpsilon(epsilon_);
    bn_op.SetTraining(is_training_);
    MUSA_KERNEL_TRACE_END("Set Attr");

    MUSA_KERNEL_TRACE_START("Kernel");
    mStatus status = bn_op.RunBwd(
        handle, mt_dx, mt_d_mean, mt_d_var, mt_d_scale, mt_d_offset, mt_x,
        mt_dy, mt_saved_mean, mt_saved_var, mt_scale, maintainer);
    MUSA_KERNEL_TRACE_END("Kernel");

    if (status == mStatus::SUCCESS && !is_nhwc_) {
      MUSA_KERNEL_TRACE_START("Permute Out");
      OP_REQUIRES_OK(ctx,
                     PermuteTensorOnMusa(ctx, dx_nhwc, dx, kPermNhwcToNchw));
      MUSA_KERNEL_TRACE_END("Permute Out");
    }

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA BN Backward failed."));
  }

 private:
  float epsilon_;
  bool is_training_;
  bool is_nhwc_;
};

#define REGISTER_MUSA_BN_ALL(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNorm").Device("MUSA").TypeConstraint<TYPE>("T"),       \
      MusaFusedBatchNormOp<TYPE>);                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNormV2").Device("MUSA").TypeConstraint<TYPE>("T"),     \
      MusaFusedBatchNormOp<TYPE>);                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNormV3").Device("MUSA").TypeConstraint<TYPE>("T"),     \
      MusaFusedBatchNormOp<TYPE>);                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNormGrad").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      MusaFusedBatchNormGradOp<TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNormGradV2").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFusedBatchNormGradOp<TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("FusedBatchNormGradV3").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFusedBatchNormGradOp<TYPE>);

REGISTER_MUSA_BN_ALL(float);
REGISTER_MUSA_BN_ALL(Eigen::half);
REGISTER_MUSA_BN_ALL(bfloat16);

}  // namespace musa
}  // namespace tensorflow
