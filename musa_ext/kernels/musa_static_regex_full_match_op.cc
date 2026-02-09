#include <regex>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

class MusaStaticRegexFullMatchOp : public OpKernel {
 public:
  explicit MusaStaticRegexFullMatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // 1. 获取正则表达式模式 (Pattern)
    string pattern_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern_str));
    
    // 2. 预编译正则表达式
    
    try {
      // 使用 ECMAScript 标准语法 (也是 C++ 默认的)
      regex_ = std::regex(pattern_str, std::regex_constants::optimize);
    } catch (const std::regex_error& e) {
      OP_REQUIRES(ctx, false, errors::InvalidArgument("Invalid regex pattern: ", e.what()));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // 3. 获取输入
    const Tensor& input_tensor = ctx->input(0);
    const auto& input_flat = input_tensor.flat<tstring>();

    // 4. 分配输出
    // 输出是 bool 类型，形状和输入一样
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<bool>();

    const int64 N = input_tensor.NumElements();

    // 5. 执行匹配 (在 CPU 上)
    // 遍历每一个字符串，进行全匹配检查
    for (int64 i = 0; i < N; ++i) {
      const std::string& s = input_flat(i);
      // std::regex_match 要求整个字符串完全匹配 pattern
      // (与之相对的是 regex_search，只要包含就行，但这里是 FullMatch)
      bool is_match = std::regex_match(s, regex_);
      output_flat(i) = is_match;
    }
  }

 private:
  std::regex regex_;
};

// 6. 注册 Kernel
// 重点：输入是 string，输出是 bool，全部留在 CPU 内存 (.HostMemory)
#define REGISTER_MUSA_KERNEL()                        \
  REGISTER_KERNEL_BUILDER(Name("StaticRegexFullMatch")\
                              .Device("MUSA")         \
                              .HostMemory("input")    \
                              .HostMemory("output"),  \
                          MusaStaticRegexFullMatchOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
