#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../utils_op.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchGelu(const T* src, T* dst, int n, bool approximate,
                musaStream_t stream);

template <typename T>
class MusaGeluOp : public MusaOpKernel {
 public:
  explicit MusaGeluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("approximate", &approximate_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "MusaGelu");

    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    MUSA_KERNEL_TRACE_START("Mem Alloc");
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    MUSA_KERNEL_TRACE_END("Mem Alloc");

    if (input.NumElements() == 0) {
      // VLOG(1) << "MusaGeluOp::Compute skipped empty tensor";
      return;
    }

    const T* input_ptr = input.flat<T>().data();
    T* output_ptr = output->flat<T>().data();
    const int64 num_elements = input.NumElements();

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    // VLOG(1) << "MusaGeluOp::Compute launching kernel, elements="
    //         << num_elements << ", approximate=" << approximate_;

    MUSA_KERNEL_TRACE_START("Kernel");
    LaunchGelu<T>(input_ptr, output_ptr, static_cast<int>(num_elements),
                  approximate_, stream);
    MUSA_KERNEL_TRACE_END("Kernel");

    // VLOG(1) << "MusaGeluOp::Compute finished, elements=" << num_elements
    //         << ", approximate=" << approximate_;
  }

 private:
  bool approximate_;
};

#define REGISTER_MUSA_GELU(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaGelu").Device("MUSA").TypeConstraint<TYPE>("T"),        \
      MusaGeluOp<TYPE>);

REGISTER_MUSA_GELU(float);
REGISTER_MUSA_GELU(double);
REGISTER_MUSA_GELU(Eigen::half);
REGISTER_MUSA_GELU(bfloat16);

#undef REGISTER_MUSA_GELU

}  // namespace musa

REGISTER_OP("MusaGelu")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("approximate: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
