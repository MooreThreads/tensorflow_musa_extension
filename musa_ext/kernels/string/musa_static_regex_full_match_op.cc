#include <regex>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {

class MusaStaticRegexFullMatchOp : public OpKernel {
 public:
  explicit MusaStaticRegexFullMatchOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string pattern_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern_str));

    try {
      regex_ = std::regex(pattern_str, std::regex_constants::optimize);
    } catch (const std::regex_error& e) {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Invalid regex pattern: ", e.what()));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const auto& input_flat = input_tensor.flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<bool>();

    const int64 N = input_tensor.NumElements();

    for (int64 i = 0; i < N; ++i) {
      const std::string& s = input_flat(i);
      bool is_match = std::regex_match(s, regex_);
      output_flat(i) = is_match;
    }
  }

 private:
  std::regex regex_;
};

#define REGISTER_MUSA_KERNEL()                         \
  REGISTER_KERNEL_BUILDER(Name("StaticRegexFullMatch") \
                              .Device("MUSA")          \
                              .HostMemory("input")     \
                              .HostMemory("output"),   \
                          MusaStaticRegexFullMatchOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
