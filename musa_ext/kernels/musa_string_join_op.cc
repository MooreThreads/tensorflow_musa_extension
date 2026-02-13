#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

class MusaStringJoinOp : public OpKernel {
 public:
  explicit MusaStringJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    const int num_inputs = inputs.size();
    if (num_inputs == 0) return;

    const Tensor& input0 = inputs[0];
    const int64 num_elements = input0.NumElements();

    for (int i = 1; i < num_inputs; ++i) {
      OP_REQUIRES(
          ctx, inputs[i].shape() == input0.shape(),
          errors::InvalidArgument("MUSA StringJoin currently requires all "
                                  "inputs to have the same shape."));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));

    auto output_flat = output->flat<tstring>();

    for (int64 i = 0; i < num_elements; ++i) {
      tstring& out_s = output_flat(i);
      out_s = inputs[0].flat<tstring>()(i);

      for (int j = 1; j < num_inputs; ++j) {
        if (!separator_.empty()) {
          out_s.append(separator_);
        }
        out_s.append(inputs[j].flat<tstring>()(i));
      }
    }
  }

 private:
  string separator_;
};

#define REGISTER_MUSA_KERNEL()                       \
  REGISTER_KERNEL_BUILDER(Name("StringJoin")         \
                              .Device("MUSA")        \
                              .HostMemory("inputs")  \
                              .HostMemory("output"), \
                          MusaStringJoinOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
