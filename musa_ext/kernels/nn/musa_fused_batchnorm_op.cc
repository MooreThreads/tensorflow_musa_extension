#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

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

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);
    const Tensor& est_mean = ctx->input(3);
    const Tensor& est_var = ctx->input(4);

    const int64_t channel_elems = scale.NumElements();
    const size_t channel_bytes = channel_elems * sizeof(float);

    Tensor* y = nullptr;
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

    auto* device = GetDeviceByCtx(ctx);
    auto& handle = device->mudnn_handle();
    auto stream = device->GetStream();
    handle.SetAllowTF32(false);

    std::vector<Tensor> workspace_holder;
    workspace_holder.reserve(4); 

    auto internal_maintainer = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      Tensor temp;
      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      workspace_holder.push_back(std::move(temp));
      return ::musa::dnn::MemoryHandler(
          workspace_holder.back().flat<uint8_t>().data(), [](void*) {});
    };
    auto maintainer = device->GetMemMaintainer(internal_maintainer);

    mFormat data_fmt = is_nhwc_ ? mFormat::NHWC : mFormat::NCHW;

    mTensor mt_x = CreateMTensor(x, data_fmt);
    mTensor mt_y = CreateMTensor(*y, data_fmt);

    mTensor mt_scale = CreateMTensor(scale, mFormat::NCHW);
    mTensor mt_offset = CreateMTensor(offset, mFormat::NCHW);
    mTensor mt_fresh_mean = CreateMTensor(*saved_mean, mFormat::NCHW);
    mTensor mt_fresh_var = CreateMTensor(*saved_var, mFormat::NCHW);

    mBatchNorm bn_op;
    bn_op.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL);
    bn_op.SetEpsilon(epsilon_);
    bn_op.SetTraining(is_training_);

    mStatus status;
    if (is_training_) {
      Tensor temp_acc_buf;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(
          DT_FLOAT, TensorShape({channel_elems * 2}), &temp_acc_buf));

      float* acc_buf_ptr = temp_acc_buf.flat<float>().data();
      float* acc_mean_ptr = acc_buf_ptr;
      float* acc_var_ptr = acc_buf_ptr + channel_elems;

      musaMemsetAsync(acc_buf_ptr, 0, channel_bytes * 2, stream);

      Tensor acc_mean_view;
      {
        Tensor slice_base = temp_acc_buf.Slice(0, channel_elems);
        OP_REQUIRES(ctx,
            acc_mean_view.CopyFrom(slice_base, scale.shape()),
            errors::Internal("Failed to create acc_mean_view from buffer"));
      }

      Tensor acc_var_view;
      {
        Tensor slice_base = temp_acc_buf.Slice(channel_elems, channel_elems * 2);
        OP_REQUIRES(ctx,
            acc_var_view.CopyFrom(slice_base, scale.shape()),
            errors::Internal("Failed to create acc_var_view from buffer"));
      }

      mTensor mt_acc_mean = CreateMTensor(acc_mean_view, mFormat::NCHW);
      mTensor mt_acc_var = CreateMTensor(acc_var_view, mFormat::NCHW);

      status =
          bn_op.RunComposite(handle, mt_y, mt_x, mt_acc_mean, mt_acc_var,
                             mt_fresh_mean, mt_fresh_var, mt_scale, mt_offset,
                             (double)exp_avg_factor_, maintainer);

      if (status == mStatus::SUCCESS) {
        float* bm_ptr = batch_mean->flat<float>().data();
        float* bv_ptr = batch_var->flat<float>().data();
        float* sm_ptr = saved_mean->flat<float>().data();
        float* sv_ptr = saved_var->flat<float>().data();
        if (sv_ptr == sm_ptr + channel_elems &&
            bv_ptr == bm_ptr + channel_elems) {
          musaMemcpyAsync(bm_ptr, sm_ptr, channel_bytes * 2,
                          musaMemcpyDeviceToDevice, stream);
        } else {
          musaMemcpyAsync(bm_ptr, sm_ptr, channel_bytes,
                          musaMemcpyDeviceToDevice, stream);
          musaMemcpyAsync(bv_ptr, sv_ptr, channel_bytes,
                          musaMemcpyDeviceToDevice, stream);
        }
      }

    } else {
      mTensor mt_est_mean = CreateMTensor(est_mean, mFormat::NCHW);
      mTensor mt_est_var = CreateMTensor(est_var, mFormat::NCHW);
      status = bn_op.RunPure(handle, mt_y, mt_x, mt_est_mean, mt_est_var,
                             mt_scale, mt_offset);
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

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dy = ctx->input(0);
    const Tensor& x = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& saved_mean = ctx->input(3);
    const Tensor& saved_var = ctx->input(4);

    const int64_t channel_elems = scale.NumElements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    Tensor* d_scale = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, scale.shape(), &d_scale));
    Tensor* d_offset = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, scale.shape(), &d_offset));
    Tensor* d_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, scale.shape(), &d_mean));
    Tensor* d_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, scale.shape(), &d_var));

    auto* device = GetDeviceByCtx(ctx);
    auto& handle = device->mudnn_handle();
    auto stream = device->GetStream();
    handle.SetAllowTF32(false);

    float* d_scale_ptr = d_scale->flat<float>().data();
    float* d_offset_ptr = d_offset->flat<float>().data();
    float* d_mean_ptr = d_mean->flat<float>().data();
    float* d_var_ptr = d_var->flat<float>().data();
    const size_t channel_bytes = channel_elems * sizeof(float);

    if (d_offset_ptr == d_scale_ptr + channel_elems &&
        d_mean_ptr == d_offset_ptr + channel_elems &&
        d_var_ptr == d_mean_ptr + channel_elems) {
      musaMemsetAsync(d_scale_ptr, 0, channel_bytes * 4, stream);
    } else {
      musaMemsetAsync(d_scale_ptr, 0, channel_bytes, stream);
      musaMemsetAsync(d_offset_ptr, 0, channel_bytes, stream);
      musaMemsetAsync(d_mean_ptr, 0, channel_bytes, stream);
      musaMemsetAsync(d_var_ptr, 0, channel_bytes, stream);
    }

    std::vector<Tensor> workspace_holder;
    workspace_holder.reserve(4);

    auto maintainer_func = [&](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      Tensor temp;
      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      workspace_holder.push_back(std::move(temp));
      return ::musa::dnn::MemoryHandler(
          workspace_holder.back().flat<uint8_t>().data(), [](void*) {});
    };
    auto maintainer = device->GetMemMaintainer(maintainer_func);

    mFormat data_fmt = is_nhwc_ ? mFormat::NHWC : mFormat::NCHW;

    mTensor mt_dy = CreateMTensor(dy, data_fmt);
    mTensor mt_x = CreateMTensor(x, data_fmt);
    mTensor mt_dx = CreateMTensor(*dx, data_fmt);

    mTensor mt_scale = CreateMTensor(scale, mFormat::NCHW);
    mTensor mt_saved_mean = CreateMTensor(saved_mean, mFormat::NCHW);
    mTensor mt_saved_var = CreateMTensor(saved_var, mFormat::NCHW);

    mTensor mt_d_scale = CreateMTensor(*d_scale, mFormat::NCHW);
    mTensor mt_d_offset = CreateMTensor(*d_offset, mFormat::NCHW);
    mTensor mt_d_mean = CreateMTensor(*d_mean, mFormat::NCHW);
    mTensor mt_d_var = CreateMTensor(*d_var, mFormat::NCHW);

    mBatchNorm bn_op;
    bn_op.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL);
    bn_op.SetEpsilon(epsilon_);
    bn_op.SetTraining(is_training_);

    mStatus status = bn_op.RunBwd(
        handle, mt_dx, mt_d_mean, mt_d_var, mt_d_scale, mt_d_offset, mt_x,
        mt_dy, mt_saved_mean, mt_saved_var, mt_scale, maintainer);

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