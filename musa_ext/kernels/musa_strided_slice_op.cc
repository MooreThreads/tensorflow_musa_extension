/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved. */



#include "utils_op.h" 

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/framework/ops_util.h"

#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/core/util/strided_slice_op.h"

#include <mudnn.h>



namespace tensorflow {

namespace musa {

namespace {



template <typename T>

class MusaStridedSliceOp : public OpKernel {

 public:

  explicit MusaStridedSliceOp(OpKernelConstruction* context) : OpKernel(context) {

  

    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask_));

    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask_));

    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask_));

    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask_));

    OP_REQUIRES_OK(context, context->GetAttr("shrink_axis_mask", &shrink_axis_mask_));

  }



  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    // 1. 解析参数
    PartialTensorShape processing_shape, final_shape;
    bool is_identity = true, slice_dim0 = true, is_simple_slice = true;
    gtl::InlinedVector<long long, 4> begin, end, strides;

    OP_REQUIRES_OK(context,
                   ::tensorflow::ValidateStridedSliceOp(
                       &context->input(1), &context->input(2), context->input(3),
                       input.shape(), begin_mask_, end_mask_, ellipsis_mask_,
                       new_axis_mask_, shrink_axis_mask_, &processing_shape,
                       &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                       &begin, &end, &strides));

    // 2. Identity 快速路径
    if (is_identity) {
      TensorShape final_tensor_shape;
      final_shape.AsTensorShape(&final_tensor_shape);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, final_tensor_shape, &output));
      if (input.NumElements() > 0) {
          auto& h = GetHandleByCtx(context);
          musaMemcpyAsync(output->flat<T>().data(), input.flat<T>().data(), 
                          input.TotalBytes(), musaMemcpyDeviceToDevice, 
                          reinterpret_cast<musaStream_t>(h.GetStream()));
      }
      return;
    }

    // 3. 分配正常输出
    TensorShape final_tensor_shape;
    final_shape.AsTensorShape(&final_tensor_shape);
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, final_tensor_shape, &result));
    
    if (result->NumElements() == 0 || input.NumElements() == 0) return;

    // 4. 构建 muDNN Tensor
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*result);

    // 设置输出 NdInfo 为处理形状 (Requirement #2)
    TensorShape proc_tensor_shape;
    processing_shape.AsTensorShape(&proc_tensor_shape);
    std::vector<int64_t> out_dims_aligned;
    for(int i=0; i<proc_tensor_shape.dims(); ++i) {
        out_dims_aligned.push_back(proc_tensor_shape.dim_size(i));
    }
    if (out_dims_aligned.empty()) out_dims_aligned.push_back(1);
    
    // 显式强制转换解决 Ambiguous Call 编译问题
    MTOP_CHECK_OK(out_mt.SetNdInfo(static_cast<int>(out_dims_aligned.size()), 
                                   out_dims_aligned.data()),
                  "SetNdInfo Out", context);

    // 5. 【核心修复】：参数长度严格锁定在 input_dims
    std::vector<int64_t> m_starts(input_dims, 0);
    std::vector<int64_t> m_strides(input_dims, 1);

    for (int i = 0; i < input_dims; ++i) {
        if (i < (int)begin.size()) {
            int64_t s_begin = static_cast<int64_t>(begin[i]);
            int64_t s_stride = static_cast<int64_t>(strides[i]);
            int64_t dim_max = input.dim_size(i);

            // 像 ReverseV2 一样对起始点进行绝对物理对齐
            if (s_stride < 0) {
                // 如果是逆序且逻辑起点超限，强制回到最后一个物理元素
                if (s_begin >= dim_max) s_begin = dim_max - 1;
                if (s_begin < 0) s_begin += dim_max;
            } else {
                // 正向寻址：禁止越界到 dim_max 导致硬件寻址非法
                if (s_begin < 0) s_begin = 0;
                if (s_begin >= dim_max && dim_max > 0) s_begin = dim_max - 1;
            }
            m_starts[i] = s_begin;
            m_strides[i] = s_stride;
        }
    }

    // 6. 执行计算与同步检查
    mHandle& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;

    MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, m_starts.data(), m_strides.data()),
                  "ConfigDimStride", context);

//mStatus status = op.Run(h, out_mt, in_mt);
   
	MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "RunOp", context);

    // 【调试关键】：强制同步，如果是这里崩了，我们能通过上面的 fprintf 拿到参数
    musaStreamSynchronize(reinterpret_cast<musaStream_t>(h.GetStream()));

  //  if (status != mStatus::SUCCESS) {
    //    context->SetStatus(errors::Internal("MUSA StridedSlice Run failed."));
   // }
  }



 private:

  // 【修正】：确保这里的变量名与 Compute 函数中调用的 begin_mask_ 等一致

  int32 begin_mask_, end_mask_;

  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;

};



// 这里的参数必须是 T (或者你统一下面用 type)
#define REGISTER_STRIDED_SLICE_MUSA(T)                          \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                  \
                              .Device("MUSA")                   \
                              .TypeConstraint<T>("T")           \
                              .HostMemory("begin")              \
                              .HostMemory("end")                \
                              .HostMemory("strides"),           \
                          MusaStridedSliceOp<T>)

// 下面注册时会自动替换 T 为具体的类型
REGISTER_STRIDED_SLICE_MUSA(float);
REGISTER_STRIDED_SLICE_MUSA(int32);
REGISTER_STRIDED_SLICE_MUSA(int64);
REGISTER_STRIDED_SLICE_MUSA(Eigen::half);
REGISTER_STRIDED_SLICE_MUSA(Eigen::bfloat16);
REGISTER_STRIDED_SLICE_MUSA(bool);



#undef REGISTER_STRIDED_SLICE_MUSA



}  // namespace

}  // namespace musa

}  // namespace tensorflow













