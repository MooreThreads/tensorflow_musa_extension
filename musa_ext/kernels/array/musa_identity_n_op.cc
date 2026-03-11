#include "../utils_op.h"

namespace tensorflow {
namespace musa {
class MusaIdentityNOp : public OpKernel {
 public:
  explicit MusaIdentityNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList input;
    OpOutputList output;

    OP_REQUIRES_OK(context, context->input_list("input", &input));
    OP_REQUIRES_OK(context, context->output_list("output", &output));
    OP_REQUIRES(context, input.size() == output.size(),
                errors::InvalidArgument("Input and output counts must match."));
    // Skip checking for TPU here and conduct `IdentityN` directly.
    for (int i = 0; i < input.size(); ++i) {
      output.set(i, input[i]);
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("IdentityN").Device("MUSA"), MusaIdentityNOp);

}  // namespace musa
}  // namespace tensorflow
