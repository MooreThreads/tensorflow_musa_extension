#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchErf(const T* src, T* dst, int n, musaStream_t stream);

template <typename T>
void LaunchErfSpecialCaseFixup(const T* src, T* dst, int n,
                               musaStream_t stream);

template <typename T>
struct ErfRunner {
  static void Run(OpKernelContext* ctx, const Tensor& input, Tensor* output,
                  mFormat format) {
    auto& handle = GetHandleByCtx(ctx);
    mTensor t_input = CreateMTensor(input, format);
    mTensor t_output = CreateMTensor(*output, format);

    mUnary op;
    op.SetMode(::musa::dnn::Unary::Mode::ERF);
    auto status = op.Run(handle, t_output, t_input);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA muDNN Unary Erf execution failed. Status: ",
                         static_cast<int>(status)));

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchErfSpecialCaseFixup<T>(input.flat<T>().data(),
                                 output->flat<T>().data(),
                                 static_cast<int>(input.NumElements()), stream);

    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA Erf special-case fixup failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

template <>
struct ErfRunner<double> {
  static void Run(OpKernelContext* ctx, const Tensor& input, Tensor* output,
                  mFormat /*format*/) {
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchErf<double>(input.flat<double>().data(),
                      output->flat<double>().data(),
                      static_cast<int>(input.NumElements()), stream);

    auto kernel_status = musaGetLastError();
    OP_REQUIRES(ctx, kernel_status == musaSuccess,
                errors::Internal("MUSA Erf kernel failed: ",
                                 musaGetErrorString(kernel_status)));
  }
};

template <typename T>
class MusaErfOp : public MusaOpKernel {
 public:
  explicit MusaErfOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  // Erf is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);

    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    ErfRunner<T>::Run(ctx, input, output, format_);
  }
};

#define REGISTER_MUSA_ERF(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Erf").Device("MUSA").TypeConstraint<TYPE>("T"), MusaErfOp<TYPE>)

REGISTER_MUSA_ERF(float);
REGISTER_MUSA_ERF(double);
REGISTER_MUSA_ERF(Eigen::half);
REGISTER_MUSA_ERF(bfloat16);

#undef REGISTER_MUSA_ERF

}  // namespace musa
}  // namespace tensorflow
