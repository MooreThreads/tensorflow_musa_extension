/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSliceOp : public MusaOpKernel {
 public:
  explicit MusaSliceOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& begin_tensor = ctx->input(1);
    const Tensor& size_tensor = ctx->input(2);

    const int dims = input.dims();
    std::vector<int64_t> starts_mt(dims);
    TensorShape output_shape;


    auto get_index_value = [](const Tensor& t, int i) -> int64_t {
      if (t.dtype() == DT_INT32) {
        return static_cast<int64_t>(t.flat<int32>()(i));
      } else {
        return static_cast<int64_t>(t.flat<int64_t>()(i));
      }
    };

    for (int i = 0; i < dims; ++i) {
      int64_t b = get_index_value(begin_tensor, i);
      int64_t s = get_index_value(size_tensor, i);

      // 处理 TensorFlow 的 -1 约定：代表取到该维度的末尾
      if (s == -1) {
        s = input.dim_size(i) - b;
      }
      
      starts_mt[i] = b;
      output_shape.AddDim(s);
    }

    // 分配输出张量
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // 准备 muDNN 句柄和张量描述符
    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*output);

    // 配置 muDNN Slice 逻辑 (通常在 Permute 类中)
    ::musa::dnn::Permute op;
    
    // 此时 starts_mt.data() 是 int64_t* 类型，符合 muDNN API 要求
    auto status = op.ConfigDimStrideForSlice(out_mt, in_mt, starts_mt.data());
    OP_REQUIRES(ctx, status == mStatus::SUCCESS, 
                errors::Internal("muDNN ConfigDimStrideForSlice failed. "
                                 "Check if input/output dims are consistent."));

    // 执行切片
    status = op.Run(handle, out_mt, in_mt);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS, 
                errors::Internal("muDNN Slice Run failed"));
  }
};

// =====================================================================
// 算子注册 (支持 6 种基础数据类型)
// =====================================================================

#define REGISTER_MUSA_SLICE(type)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Slice").Device(DEVICE_MTGPU).TypeConstraint<type>("T")     \
      .HostMemory("begin")                                             \
      .HostMemory("size"),                                             \
      MusaSliceOp<type>);

REGISTER_MUSA_SLICE(float);          // FP32
REGISTER_MUSA_SLICE(double);         // FP64
REGISTER_MUSA_SLICE(int32);          // INT32
REGISTER_MUSA_SLICE(int64);          // INT64
REGISTER_MUSA_SLICE(Eigen::half);    // FP16
REGISTER_MUSA_SLICE(bfloat16);       // BF16

} // namespace musa
} // namespace tensorflow
