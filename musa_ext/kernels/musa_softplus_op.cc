#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSoftplusOp : public MusaOpKernel {
 public:
  explicit MusaSoftplusOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    Tensor t_abs_tf, t_neg_abs_tf, t_exp_tf, t_exp_add1_tf, t_log_tf, t_relu_tf;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_abs_tf));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_neg_abs_tf));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_exp_tf));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_exp_add1_tf));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_log_tf));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &t_relu_tf));

    ::musa::dnn::Tensor x      = CreateMTensor(input, format_);
    ::musa::dnn::Tensor y      = CreateMTensor(*output, format_);
    ::musa::dnn::Tensor t_abs  = CreateMTensor(t_abs_tf, format_);
    ::musa::dnn::Tensor t_nabs = CreateMTensor(t_neg_abs_tf, format_);
    ::musa::dnn::Tensor t_exp  = CreateMTensor(t_exp_tf, format_);
    ::musa::dnn::Tensor t_e1   = CreateMTensor(t_exp_add1_tf, format_);
    ::musa::dnn::Tensor t_log  = CreateMTensor(t_log_tf, format_);
    ::musa::dnn::Tensor t_relu = CreateMTensor(t_relu_tf, format_);

    auto check_status = [&](::musa::dnn::Status status, const char* msg) {
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal(msg, " Status: ", static_cast<int>(status)));
    };

    // 1) t_abs = abs(x)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::ABS),
                   "MUSA Softplus ABS SetMode failed.");
      auto status = op.Run(handle, t_abs, x);
      check_status(status, "MUSA Softplus ABS execution failed.");
    }

    // 2) t_neg_abs = -t_abs   (via unary MUL with alpha = -1)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::MUL),
                   "MUSA Softplus MUL(SetMode) failed.");
      check_status(op.SetAlpha(-1.0),
                   "MUSA Softplus MUL(SetAlpha=-1) failed.");
      auto status = op.Run(handle, t_nabs, t_abs);
      check_status(status, "MUSA Softplus MUL(-1) execution failed.");
    }

    // 3) t_exp = exp(t_neg_abs)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::EXP),
                   "MUSA Softplus EXP SetMode failed.");
      auto status = op.Run(handle, t_exp, t_nabs);
      check_status(status, "MUSA Softplus EXP execution failed.");
    }

    // 4) t_exp_add1 = t_exp + 1   (via unary ADD with alpha = 1)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::ADD),
                   "MUSA Softplus ADD(SetMode) failed.");
      check_status(op.SetAlpha(1.0),
                   "MUSA Softplus ADD(SetAlpha=1) failed.");
      auto status = op.Run(handle, t_e1, t_exp);
      check_status(status, "MUSA Softplus ADD(+1) execution failed.");
    }

    // 5) t_log = log(t_exp_add1)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::LOG),
                   "MUSA Softplus LOG SetMode failed.");
      auto status = op.Run(handle, t_log, t_e1);
      check_status(status, "MUSA Softplus LOG execution failed.");
    }

    // 6) t_relu = relu(x)   == max(x, 0)
    {
      ::musa::dnn::Unary op;
      check_status(op.SetMode(::musa::dnn::Unary::Mode::RELU),
                   "MUSA Softplus RELU SetMode failed.");
      auto status = op.Run(handle, t_relu, x);
      check_status(status, "MUSA Softplus RELU execution failed.");
    }

    // 7) y = t_relu + t_log
    {
      ::musa::dnn::Binary op;
      check_status(op.SetMode(::musa::dnn::Binary::Mode::ADD),
                   "MUSA Softplus Binary ADD SetMode failed.");
      auto status = op.Run(handle, y, t_relu, t_log);
      check_status(status, "MUSA Softplus final ADD execution failed.");
    }
  }
};

#define REGISTER_MUSA_SOFTPLUS(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Softplus").Device("MUSA").TypeConstraint<TYPE>("T"),          \
      MusaSoftplusOp<TYPE>)

REGISTER_MUSA_SOFTPLUS(float);
REGISTER_MUSA_SOFTPLUS(Eigen::half);
REGISTER_MUSA_SOFTPLUS(bfloat16);

#undef REGISTER_MUSA_SOFTPLUS

}  // namespace musa
}  // namespace tensorflow