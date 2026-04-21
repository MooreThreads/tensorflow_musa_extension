#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

class MusaCastOp : public MusaOpKernel {
 public:
  explicit MusaCastOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));
    // Cache identity check for zero-copy fast path (matches TensorFlow's CastOpBase)
    is_identity_cast_ = (external_src_dtype_ == external_dst_dtype_);
  }

  // Cast is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& inp = ctx->input(0);

    // Zero-copy fast path for identity cast (SrcT == DstT)
    // This matches TensorFlow's official CastOpBase behavior:
    // - Uses reference counting to manage shared buffer
    // - TensorFlow's runtime ensures safety via copy-on-write semantics
    if (is_identity_cast_) {
      ctx->set_output(0, inp);
      return;
    }

    // Early exit for empty tensors - still need to allocate output with correct shape
    if (inp.NumElements() == 0) {
      ctx->set_output(0, inp);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &output));

    auto in_mt = CreateMTensor(inp);
    auto out_mt = CreateMTensor(*output);

    // BOOL format workaround for muDNN
    if (inp.dtype() == DT_BOOL) {
      in_mt.SetFormat(mFormat::NCHW);
    }

    mHandle& h = GetHandleByCtx(ctx);
    ::musa::dnn::Unary op;

    auto m_status = op.SetMode(::musa::dnn::Unary::Mode::CAST);
    OP_REQUIRES(ctx, m_status == mStatus::SUCCESS,
                errors::Internal("muDNN Unary SetMode failed in Cast"));

    m_status = op.Run(h, out_mt, in_mt);

    if (m_status != mStatus::SUCCESS) {
      LOG(ERROR) << "MUSA Cast Run failed! Src: "
                 << DataTypeString(external_src_dtype_)
                 << " -> Dst: " << DataTypeString(external_dst_dtype_)
                 << " | Status: " << static_cast<int>(m_status);

      ctx->SetStatus(errors::Internal("MUSA Cast Run failed. Status code: ",
                                      static_cast<int>(m_status)));
      return;
    }
  }

 private:
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
  bool is_identity_cast_;  // Cached flag for zero-copy optimization
};

#define REGISTER_CAST_MUSA(SrcT, DstT)                       \
  REGISTER_KERNEL_BUILDER(Name("Cast")                       \
                              .Device(DEVICE_MTGPU)          \
                              .TypeConstraint<SrcT>("SrcT")  \
                              .TypeConstraint<DstT>("DstT"), \
                          MusaCastOp);

REGISTER_CAST_MUSA(bool, bool);
REGISTER_CAST_MUSA(bool, int32);
REGISTER_CAST_MUSA(bool, int64);
REGISTER_CAST_MUSA(bool, Eigen::half);
REGISTER_CAST_MUSA(bool, bfloat16);
REGISTER_CAST_MUSA(bool, float);
REGISTER_CAST_MUSA(bool, double);

REGISTER_CAST_MUSA(int32, bool);
REGISTER_CAST_MUSA(int32, int32);
REGISTER_CAST_MUSA(int32, int64);
REGISTER_CAST_MUSA(int32, Eigen::half);
REGISTER_CAST_MUSA(int32, bfloat16);
REGISTER_CAST_MUSA(int32, float);
REGISTER_CAST_MUSA(int32, double);

REGISTER_CAST_MUSA(int64, bool);
REGISTER_CAST_MUSA(int64, int32);
REGISTER_CAST_MUSA(int64, int64);
REGISTER_CAST_MUSA(int64, Eigen::half);
REGISTER_CAST_MUSA(int64, bfloat16);
REGISTER_CAST_MUSA(int64, float);
REGISTER_CAST_MUSA(int64, double);

REGISTER_CAST_MUSA(Eigen::half, bool);
REGISTER_CAST_MUSA(Eigen::half, int32);
REGISTER_CAST_MUSA(Eigen::half, int64);
REGISTER_CAST_MUSA(Eigen::half, Eigen::half);
REGISTER_CAST_MUSA(Eigen::half, bfloat16);
REGISTER_CAST_MUSA(Eigen::half, float);
REGISTER_CAST_MUSA(Eigen::half, double);

REGISTER_CAST_MUSA(bfloat16, bool);
REGISTER_CAST_MUSA(bfloat16, int32);
REGISTER_CAST_MUSA(bfloat16, int64);
REGISTER_CAST_MUSA(bfloat16, Eigen::half);
REGISTER_CAST_MUSA(bfloat16, bfloat16);
REGISTER_CAST_MUSA(bfloat16, float);
REGISTER_CAST_MUSA(bfloat16, double);

REGISTER_CAST_MUSA(float, bool);
REGISTER_CAST_MUSA(float, int32);
REGISTER_CAST_MUSA(float, int64);
REGISTER_CAST_MUSA(float, Eigen::half);
REGISTER_CAST_MUSA(float, bfloat16);
REGISTER_CAST_MUSA(float, float);
REGISTER_CAST_MUSA(float, double);

REGISTER_CAST_MUSA(double, bool);
REGISTER_CAST_MUSA(double, int32);
REGISTER_CAST_MUSA(double, int64);
REGISTER_CAST_MUSA(double, Eigen::half);
REGISTER_CAST_MUSA(double, bfloat16);
REGISTER_CAST_MUSA(double, float);
REGISTER_CAST_MUSA(double, double);

}  // namespace musa
}  // namespace tensorflow
