#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "absl/strings/str_cat.h"
#include "mu/device/musa_memcpy.h"
#include "utils_op.h" 
#include <musa_runtime.h> // 为了使用 musaDeviceSynchronize

namespace tensorflow {
namespace musa {

/**
 * MusaPrintVOp: 安全地将 MUSA 数据搬回 CPU 打印
 */
class MusaPrintVOp : public OpKernel {
 public:
  explicit MusaPrintVOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* c) override {
  
    musaDeviceSynchronize();

    for (int i = 0; i < c->num_inputs(); ++i) {
        const Tensor& input = c->input(i);
        
        // 2. 准备 CPU 容器
        Tensor cpu_tensor(input.dtype(), input.shape());
        
        // 3. 执行 D2H 拷贝 (显存 -> 内存)
        // 假设你的 MusaMemcpyD2H 定义为: (void* dst, const void* src, size_t size)
        MusaMemcpyD2H(cpu_tensor.data(), input.data(), input.TotalBytes());

        // 4. 打印内容
        std::string s = cpu_tensor.SummarizeValue(summarize_);
        std::cerr << ">>> [MUSA_PRINT] Input " << i << ": " << s << std::endl;
    }
  }
 private:
  int summarize_;
};

/**
 * MusaStringFormatOp: 同样的同步拷贝逻辑
 */
class MusaStringFormatOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(OpKernelContext* c) override {
    musaDeviceSynchronize();
    
    std::string result = "MUSA_FORMAT: ";
    for (int i = 0; i < c->num_inputs(); ++i) {
      const Tensor& input = c->input(i);
      Tensor cpu_tensor(input.dtype(), input.shape());
      MusaMemcpyD2H(cpu_tensor.data(), input.data(), input.TotalBytes());
      
      absl::StrAppend(&result, "[Input ", i, "]: ", cpu_tensor.SummarizeValue(3), " ");
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<tstring>()() = result;
  }
};

// 注册
REGISTER_KERNEL_BUILDER(Name("PrintV").Device("MUSA"), MusaPrintVOp);
REGISTER_KERNEL_BUILDER(Name("StringFormat").Device("MUSA"), MusaStringFormatOp);

} // namespace musa
} // namespace tensorflow

