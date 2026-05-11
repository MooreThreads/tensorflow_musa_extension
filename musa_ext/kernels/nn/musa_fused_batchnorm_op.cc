#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

// Bessel de-correction kernel: converts muDNN sample variance (1/(N-1)) back
// to population variance (1/N) to match TF CPU FusedBatchNorm output
// convention.
extern "C" {
void LaunchBesselCorrection(float* data, float factor, int count,
                            musaStream_t stream);
}

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

  // BatchNorm is computationally intensive (reduction operations)
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);
    const Tensor& est_mean = ctx->input(3);
    const Tensor& est_var = ctx->input(4);

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
    // CPU FusedBatchNormV3 allocates reserve_3 as a 0-D scalar TensorShape({}).
    // muDNN does not populate this field; zero-initialise for safety.
    OP_REQUIRES_OK(ctx, ctx->allocate_output(5, TensorShape({}), &reserve_3));

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

    // Zero-initialise reserve_3 so it is always in a defined state.
    // The current backward op (MusaFusedBatchNormGradOp) does not consume
    // reserve_3, but TF's graph executor may inspect its shape/value.
    musaMemsetAsync(reserve_3->flat<float>().data(), 0, reserve_3->TotalBytes(),
                    stream);

    mStatus status;
    if (is_training_) {
      Tensor temp_acc_mean, temp_acc_var;
      ctx->allocate_temp(DT_FLOAT, scale.shape(), &temp_acc_mean);
      ctx->allocate_temp(DT_FLOAT, scale.shape(), &temp_acc_var);

      musaMemsetAsync(temp_acc_mean.flat<float>().data(), 0,
                      temp_acc_mean.NumElements() * sizeof(float), stream);
      musaMemsetAsync(temp_acc_var.flat<float>().data(), 0,
                      temp_acc_var.NumElements() * sizeof(float), stream);

      mTensor mt_acc_mean = CreateMTensor(temp_acc_mean, mFormat::NCHW);
      mTensor mt_acc_var = CreateMTensor(temp_acc_var, mFormat::NCHW);

      status =
          bn_op.RunComposite(handle, mt_y, mt_x, mt_acc_mean, mt_acc_var,
                             mt_fresh_mean, mt_fresh_var, mt_scale, mt_offset,
                             (double)exp_avg_factor_, maintainer);

      if (status == mStatus::SUCCESS) {
        // muDNN RunComposite writes the batch statistics (mean / sample
        // variance) into acc_mean / acc_var.  fresh_mean / fresh_var are
        // *input* parameters (the prior running stats) and are NOT populated
        // by the kernel, so reading from saved_mean/saved_var (= fresh_*)
        // would return uninitialised / stale data.
        size_t copy_size = saved_mean->NumElements() * sizeof(float);

        // batch_mean = saved_mean = batch mean from acc_mean
        musaMemcpyAsync(batch_mean->flat<float>().data(),
                        temp_acc_mean.flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);
        musaMemcpyAsync(saved_mean->flat<float>().data(),
                        temp_acc_mean.flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);

        // muDNN RunComposite writes *sample* variance (1/(N-1)) into acc_var.
        // TF CPU FusedBatchNorm convention:
        //   batch_var  (output[2]) = sample variance  (1/(N-1)),
        //   Bessel-corrected saved_var  (output[4]) = population variance
        //   (1/N), NOT Bessel-corrected
        //
        // Therefore:
        //   batch_var <- acc_var directly (muDNN already provides sample
        //   variance) saved_var <- acc_var * (N-1)/N  (convert sample →
        //   population variance)
        musaMemcpyAsync(batch_var->flat<float>().data(),
                        temp_acc_var.flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);
        musaMemcpyAsync(saved_var->flat<float>().data(),
                        temp_acc_var.flat<float>().data(), copy_size,
                        musaMemcpyDeviceToDevice, stream);

        // N_pixels = N * H * W  (elements per channel)
        int64_t N_pixels =
            is_nhwc_ ? (x.dim_size(0) * x.dim_size(1) * x.dim_size(2))   // NHWC
                     : (x.dim_size(0) * x.dim_size(2) * x.dim_size(3));  // NCHW
        if (N_pixels > 1) {
          // Factor (N-1)/N converts sample variance → population variance.
          // Applied only to saved_var; batch_var stays as sample variance.
          float correction =
              static_cast<float>(N_pixels - 1) / static_cast<float>(N_pixels);
          int C = static_cast<int>(scale.NumElements());
          LaunchBesselCorrection(saved_var->flat<float>().data(), correction, C,
                                 stream);
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

  // BatchNormGrad is computationally intensive
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dy = ctx->input(0);
    const Tensor& x = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& saved_mean = ctx->input(3);
    const Tensor& saved_var = ctx->input(4);

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

    musaMemsetAsync(const_cast<char*>(dx->tensor_data().data()), 0,
                    dx->TotalBytes(), stream);
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
