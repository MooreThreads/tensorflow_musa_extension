#include <musa_runtime.h>

#include "../utils_op.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mu/device/musa_memcpy.h"

namespace tensorflow {
namespace musa {

std::string TensorToSummary(OpKernelContext* c, const Tensor& device_tensor,
                            int summarize);

class MusaAssertOp : public MusaOpKernel {
 public:
  explicit MusaAssertOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(cond.shape()),
                errors::InvalidArgument("In[0] should be a scalar: ",
                                        cond.shape().DebugString()));

    if (cond.scalar<bool>()()) {
      return;
    }

    musaDeviceSynchronize();

    std::string msg = "assertion failed: ";
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      const Tensor& device_tensor = ctx->input(i);
      auto summarizedValue = TensorToSummary(ctx, device_tensor, summarize_);
      absl::StrAppend(&msg, "[", summarizedValue, "]");
      if (i < ctx->num_inputs() - 1) absl::StrAppend(&msg, " ");
    }
    ctx->SetStatus(errors::InvalidArgument(msg));
  }

  bool IsExpensive() override { return false; }

 private:
  int32_t summarize_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("Assert").Device("MUSA"), MusaAssertOp);

}  // namespace musa
}  // namespace tensorflow