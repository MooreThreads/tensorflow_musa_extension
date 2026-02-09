#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"
#include <mudnn.h>
#include <musa_runtime_api.h>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaLessEqualOp : public MusaOpKernel {
 public:
  explicit MusaLessEqualOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_0 = ctx->input(0);
    const Tensor& input_1 = ctx->input(1);

    // 1. 计算正确的输出形状
    BCast bcast(BCast::FromShape(input_0.shape()),
                BCast::FromShape(input_1.shape()));
    
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes: ", input_0.shape().DebugString(),
                                        " vs. ", input_1.shape().DebugString()));
    
    TensorShape output_shape = BCast::ToShape(bcast.result_shape());
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    // 2. 策略：全部展平为 1D 进行计算
   
    
    // A. 准备输入 Views (1D)
    TensorShape flat_shape_in0({input_0.NumElements()});
    TensorShape flat_shape_in1({input_1.NumElements()});
    
   
    if (input_0.shape() == input_1.shape()) {
        // --- 路径 A: 形状相同，强制 1D ---
        Tensor input_0_flat; 
        Tensor input_1_flat;
        CHECK(input_0_flat.CopyFrom(input_0, flat_shape_in0));
        CHECK(input_1_flat.CopyFrom(input_1, flat_shape_in1));
        
        // 创建一个临时的 1D 输出 Tensor
        Tensor temp_output;
        AllocatorAttributes attr;
        attr.set_on_host(false);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_BOOL, flat_shape_in0, &temp_output, attr));
        
        // 执行 muDNN
        ::musa::dnn::Tensor mt_in0 = CreateMTensor(input_0_flat, format_);
        ::musa::dnn::Tensor mt_in1 = CreateMTensor(input_1_flat, format_);
        ::musa::dnn::Tensor mt_out = CreateMTensor(temp_output, format_);
        
        ::musa::dnn::Binary binary_op;
        binary_op.SetMode(::musa::dnn::Binary::Mode::LE);
        auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);
        OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                   errors::Internal("MUSA LessEqual execution failed. Status: ", (int)status));
        
  
        auto dst_ptr = output->flat<bool>().data();
        auto src_ptr = temp_output.flat<bool>().data();
        auto copy_status = musaMemcpy(dst_ptr, src_ptr, output->NumElements() * sizeof(bool), musaMemcpyDeviceToDevice);
        OP_REQUIRES(ctx, copy_status == musaSuccess,
                    errors::Internal("MUSA memcpy failed"));

    } else {
     
        auto get_padded_4d_shape = [](const TensorShape& s) -> TensorShape {
            if (s.dims() >= 4) return s;
            TensorShape new_s = s;
            while (new_s.dims() < 4) {
                new_s.InsertDim(0, 1);
            }
            return new_s;
        };
        
        Tensor input_0_view;
        Tensor input_1_view;
        Tensor output_view; 
        CHECK(input_0_view.CopyFrom(input_0, get_padded_4d_shape(input_0.shape())));
        CHECK(input_1_view.CopyFrom(input_1, get_padded_4d_shape(input_1.shape())));
        CHECK(output_view.CopyFrom(*output, get_padded_4d_shape(output_shape)));
        
        ::musa::dnn::Tensor mt_in0 = CreateMTensor(input_0_view, format_);
        ::musa::dnn::Tensor mt_in1 = CreateMTensor(input_1_view, format_);
        ::musa::dnn::Tensor mt_out = CreateMTensor(output_view, format_);
        
        ::musa::dnn::Binary binary_op;
        binary_op.SetMode(::musa::dnn::Binary::Mode::LE);
        auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);
        OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                   errors::Internal("MUSA LessEqual Broadcast execution failed. Status: ", (int)status));
    }
  }
};

#define REGISTER_MUSA_LESS_EQUAL(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("LessEqual")                          \
                              .Device("MUSA")                        \
                              .TypeConstraint<TYPE>("T"),            \
                          MusaLessEqualOp<TYPE>)

REGISTER_MUSA_LESS_EQUAL(float);
REGISTER_MUSA_LESS_EQUAL(Eigen::half);
REGISTER_MUSA_LESS_EQUAL(bfloat16);
REGISTER_MUSA_LESS_EQUAL(int32);
REGISTER_MUSA_LESS_EQUAL(int64);

#undef REGISTER_MUSA_LESS_EQUAL

} // namespace musa
} // namespace tensorflow
