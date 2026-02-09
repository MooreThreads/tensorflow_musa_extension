/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace musa {

class MusaSqueezeOp : public OpKernel {
 public:
  explicit MusaSqueezeOp(OpKernelConstruction* c) : OpKernel(c) {
    // 获取需要压缩的轴（如果用户指定了的话）
    OP_REQUIRES_OK(c, c->GetAttr("squeeze_dims", &squeeze_dims_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);

    // 1. 计算压缩后的 Shape
    // 这里我们可以直接利用 TF 提供的工具函数，或者手动逻辑：
    // 遍历 input.shape()，去掉所有 dim_size == 1 且在 squeeze_dims_ 里的维度
    TensorShape output_shape;
    for (int i = 0; i < input.dims(); ++i) {
      bool should_squeeze = false;
      if (input.dim_size(i) == 1) {
        if (squeeze_dims_.empty()) {
          should_squeeze = true; // 默认压缩所有大小为 1 的维
        } else {
          for (int d : squeeze_dims_) {
            // 处理负索引
            int positive_d = d < 0 ? d + input.dims() : d;
            if (i == positive_d) { should_squeeze = true; break; }
          }
        }
      }
      if (!should_squeeze) {
        output_shape.AddDim(input.dim_size(i));
      }
    }

    // 2. 关键：零拷贝实现
    
    Tensor output;
    if (!output.CopyFrom(input, output_shape)) {
      c->CtxFailure(errors::Internal("Failed to squeeze tensor shape"));
      return;
    }
    
    // 3. 将结果挂载到输出
    c->set_output(0, output);
    
    std::cerr << ">>> [MUSA_DEBUG] Squeeze: " 
              << input.shape().DebugString() << " -> " 
              << output_shape.DebugString() << std::endl;
  }

 private:
  std::vector<int32> squeeze_dims_;
};

// 注册支持所有类型，因为 Squeeze 不看数据内容，只看形状
#define REGISTER_MUSA_SQUEEZE(type)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Squeeze").Device("MUSA").TypeConstraint<type>("T"), MusaSqueezeOp);

REGISTER_MUSA_SQUEEZE(float);
REGISTER_MUSA_SQUEEZE(Eigen::half);
REGISTER_MUSA_SQUEEZE(bfloat16);
REGISTER_MUSA_SQUEEZE(int32);
REGISTER_MUSA_SQUEEZE(int64);
REGISTER_MUSA_SQUEEZE(bool);
REGISTER_MUSA_SQUEEZE(double);
REGISTER_MUSA_SQUEEZE(uint8);

} // namespace musa
} // namespace tensorflow

