/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved. */

#include "utils_op.h" 
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include <mudnn.h>

namespace tensorflow {
namespace musa {
namespace {

template <typename T, typename Tdim>
class MusaExpandDimsOp : public MusaOpKernel {
 public:
  explicit MusaExpandDimsOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dim_tensor = context->input(1);
    
    // 1. 获取并校验维度索引 (dim)
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dim_tensor.shape()),
                errors::InvalidArgument("dim input must be a scalar"));
    Tdim dim = dim_tensor.scalar<Tdim>()();
    const int input_dims = input.dims();

    // 处理负数索引 (例如 -1 表示在最后增加一维)
    if (dim < 0) {
      dim += input_dims + 1;
    }

    OP_REQUIRES(context, dim >= 0 && dim <= input_dims,
                errors::InvalidArgument("Inserted dimension ", dim,
                                        " must be in range [0, ", input_dims, "]"));

    // 2. 计算输出形状
    TensorShape out_shape;
    for (int i = 0; i < dim; ++i) {
      out_shape.AddDim(input.dim_size(i));
    }
    out_shape.AddDim(1);
    for (int i = dim; i < input_dims; ++i) {
      out_shape.AddDim(input.dim_size(i));
    }

    // 3. 分配输出张量
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    if (input.NumElements() == 0) return;

    // =====================================================================
    // 4. MUSA 处理逻辑
   
    // =====================================================================
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*output);
    
    // 获取 Handle (通过基类提供的接口或 utils)
    auto& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;


    std::vector<int64_t> m_dims;
    for (int i = 0; i < input_dims; ++i) {
        m_dims.push_back(static_cast<int64_t>(input.dim_size(i)));
    }

    // 2. 如果是标量(0维)，muDNN 通常要求至少传一个 1
    if (m_dims.empty()) {
        m_dims.push_back(1);
    }

    // 3. 调用 SetNdInfo，并进行显式类型转换
    // 使用 static_cast<int> 明确指出调用哪个重载版本
    MTOP_CHECK_OK(out_mt.SetNdInfo(static_cast<int>(m_dims.size()), m_dims.data()),
                  "SetNdInfo for ExpandDims", context);
    // 【使用 MTOP_CHECK_OK_RUN】：
    // 执行数据搬运
    MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Permute Run for ExpandDims", context);
  }
};

// 注册宏定义，兼容 int32 和 int64 类型的 Tdim
#define REGISTER_MUSA_EXPAND_DIMS(type)                         \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                    \
                              .Device("MUSA")             \
                              .TypeConstraint<type>("T")        \
                              .TypeConstraint<int32>("Tdim")    \
                              .HostMemory("dim"),               \
                          MusaExpandDimsOp<type, int32>);       \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                    \
                              .Device("MUSA")             \
                              .TypeConstraint<type>("T")        \
                              .TypeConstraint<int64>("Tdim")    \
                              .HostMemory("dim"),               \
                          MusaExpandDimsOp<type, int64>);

REGISTER_MUSA_EXPAND_DIMS(float);
REGISTER_MUSA_EXPAND_DIMS(int32);
REGISTER_MUSA_EXPAND_DIMS(int64);
REGISTER_MUSA_EXPAND_DIMS(Eigen::half);
REGISTER_MUSA_EXPAND_DIMS(bool);
REGISTER_MUSA_EXPAND_DIMS(double);
REGISTER_MUSA_EXPAND_DIMS(bfloat16);
REGISTER_MUSA_EXPAND_DIMS(uint8);


#undef REGISTER_MUSA_EXPAND_DIMS

}  // namespace
}  // namespace musa
}  // namespace tensorflow

