/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include <mudnn.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "utils_op.h" 

namespace tensorflow {
namespace musa {

// =========================================================================
// 1. MusaPackOp (Stack)

// =========================================================================
template <typename T>
class MusaPackOp : public OpKernel {
 public:
  explicit MusaPackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* c) override {
    const int num = num_inputs();
    const Tensor& first_input = c->input(0);

    int expanded_num_dims = first_input.dims() + 1;
    int axis = axis_ < 0 ? axis_ + expanded_num_dims : axis_;

    TensorShape output_shape(first_input.shape());
    output_shape.InsertDim(axis, num);

    if (num == 1) {
      Tensor output;
      CHECK(output.CopyFrom(first_input, output_shape));
      c->set_output(0, output);
      return;
    }

    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    // --- 计算统一的视图维度 ---
    int64_t before_dim = 1;
    for (int i = 0; i < axis; ++i) before_dim *= output_shape.dim_size(i);

    int64_t after_dim = 1;
    for (int i = axis + 1; i < output_shape.dims(); ++i) after_dim *= output_shape.dim_size(i);

    const int64_t axis_dim = output_shape.dim_size(axis); 

    
    std::vector<int64_t> shared_input_dims = {before_dim, after_dim};

    // 构造输入列表
    std::vector<::musa::dnn::Tensor> input_tensors;
    input_tensors.reserve(num);

    for (int i = 0; i < num; ++i) {
      auto tmp_in = CreateMTensor(c->input(i));
      
      // 所有输入共享同一个 shape 数据的指针
      // 因为 shared_input_dims 定义在循环外，地址固定且有效
      tmp_in.SetNdInfo(2, shared_input_dims.data());
      
      input_tensors.push_back(tmp_in);
    }

    auto& h = GetHandleByCtx(c);
    ::musa::dnn::Concat concat;
    auto out = CreateMTensor(*output);
    
    // 输出视图
    std::vector<int64_t> out_dims = {before_dim, after_dim * axis_dim};
    out.SetNdInfo(static_cast<int>(out_dims.size()), out_dims.data());

    concat.SetAxis(1); 
    
    // 执行计算
    auto status = concat.Run(h, out, input_tensors.size(), input_tensors.data());
    OP_REQUIRES(c, status == ::musa::dnn::Status::SUCCESS, errors::Internal("MUSA Pack Run failed"));
  }

 private:
  int axis_;
};

// =========================================================================
// 2. MusaUnpackOp (Unstack)
// =========================================================================
template <typename T>
class MusaUnpackOp : public OpKernel {
 public:
  explicit MusaUnpackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);
    int axis = axis_ < 0 ? axis_ + input.dims() : axis_;
    const int num = input.dim_size(axis);

    TensorShape output_shape = input.shape();
    output_shape.RemoveDim(axis); 

    auto& h = GetHandleByCtx(c);
    auto mudnn_in = CreateMTensor(input);
    
    std::vector<int64_t> starts(input.dims(), 0);

    
    std::vector<int64_t> shared_view_dims;
    for(int d = 0; d < input.dims(); ++d) {
        shared_view_dims.push_back(input.dim_size(d));
    }
    shared_view_dims[axis] = 1;

    for (int i = 0; i < num; ++i) {
      Tensor* output;
      OP_REQUIRES_OK(c, c->allocate_output(i, output_shape, &output));
      if (output->NumElements() == 0) continue;

      auto mudnn_out = CreateMTensor(*output);
      
      // 使用循环外定义的 shared_view_dims，更安全
      mudnn_out.SetNdInfo(static_cast<int>(shared_view_dims.size()), shared_view_dims.data());

      ::musa::dnn::Permute slice_op;
      std::fill(starts.begin(), starts.end(), 0);
      starts[axis] = i;

      auto status = slice_op.ConfigDimStrideForSlice(mudnn_out, mudnn_in, starts.data());
      OP_REQUIRES(c, status == ::musa::dnn::Status::SUCCESS, 
                  errors::Internal("MUSA Unpack ConfigDimStrideForSlice failed"));

      status = slice_op.Run(h, mudnn_out, mudnn_in);
      OP_REQUIRES(c, status == ::musa::dnn::Status::SUCCESS, 
                  errors::Internal("MUSA Unpack Slice Run failed"));
    }
  }

 private:
  int axis_;
};

// --- 注册算子 ---
#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Pack").Device("MUSA").TypeConstraint<type>("T"), MusaPackOp<type>); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Unpack").Device("MUSA").TypeConstraint<type>("T"), MusaUnpackOp<type>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(double);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int64);
REGISTER_KERNELS(Eigen::half);
REGISTER_KERNELS(bfloat16);

}  // namespace musa
}  // namespace tensorflow