#include <mudnn.h>

#include <vector>

#include "mu/device/musa_memset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaBroadcastToOp : public MusaOpKernel {
 public:
  explicit MusaBroadcastToOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    if (shape_tensor.dtype() == DT_INT32) {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                              shape_tensor.flat<int32>().data(),
                              shape_tensor.NumElements(), &output_shape));
    } else if (shape_tensor.dtype() == DT_INT64) {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                              shape_tensor.flat<int64>().data(),
                              shape_tensor.NumElements(), &output_shape));
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Shape must be int32 or int64"));
    }

    if (output_shape == input_tensor.shape()) {
      ctx->set_output(0, input_tensor);
      return;
    }

    BCast bcast(BCast::FromShape(input_tensor.shape()),
                BCast::FromShape(output_shape), true);

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_tensor.shape().DebugString(),
                    " vs. ", output_shape.DebugString()));
    OP_REQUIRES(ctx, BCast::ToShape(bcast.output_shape()) == output_shape,
                errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                        input_tensor.shape(),
                                        " to tensor of shape ", output_shape));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    if (output_shape.num_elements() == 0) return;

    auto x_shape = bcast.x_reshape();
    auto y_shape = bcast.y_reshape();

    if (y_shape.size() > 8) {
      OP_REQUIRES(ctx, false,
                  errors::Unimplemented("BroadcastTo with effective rank > 8 "
                                        "is not supported by MUSA backend."));
    }

    auto& handle = GetHandleByCtx(ctx);

    Tensor zero_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input_tensor.dtype(), output_shape,
                                           &zero_tensor));

    auto status_memset =
        Memset(handle, const_cast<char*>(zero_tensor.tensor_data().data()),
               zero_tensor.TotalBytes(), 0);
    OP_REQUIRES(ctx, status_memset == mStatus::SUCCESS,
                errors::Internal("Musa Memset failed."));

    auto mt_in = CreateMTensor(input_tensor, format_);
    auto mt_out = CreateMTensor(*output_tensor, format_);
    auto mt_zero = CreateMTensor(zero_tensor, format_);

    std::vector<int64_t> in_dims(x_shape.begin(), x_shape.end());
    std::vector<int64_t> out_dims(y_shape.begin(), y_shape.end());

    OP_REQUIRES(ctx,
                mt_in.SetNdInfo(static_cast<int>(in_dims.size()),
                                in_dims.data()) == mStatus::SUCCESS,
                errors::Internal("SetNdInfo for input failed"));
    OP_REQUIRES(ctx,
                mt_out.SetNdInfo(static_cast<int>(out_dims.size()),
                                 out_dims.data()) == mStatus::SUCCESS,
                errors::Internal("SetNdInfo for output failed"));
    OP_REQUIRES(ctx,
                mt_zero.SetNdInfo(static_cast<int>(out_dims.size()),
                                  out_dims.data()) == mStatus::SUCCESS,
                errors::Internal("SetNdInfo for zero-tensor failed"));

    mBinary op;
    op.SetMode(BINARY_MODE::ADD);

    auto status = op.Run(handle, mt_out, mt_in, mt_zero);

    OP_REQUIRES(
        ctx, status == mStatus::SUCCESS,
        errors::Internal("MUSA Binary(Broadcast) execution failed. Status: ",
                         static_cast<int>(status)));
  }
};

#define REGISTER_BROADCAST_TO(type)                       \
  REGISTER_KERNEL_BUILDER(Name("BroadcastTo")             \
                              .Device("MUSA")             \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T"), \
                          MusaBroadcastToOp);

REGISTER_BROADCAST_TO(float);
REGISTER_BROADCAST_TO(double);
REGISTER_BROADCAST_TO(Eigen::half);
REGISTER_BROADCAST_TO(bfloat16);
REGISTER_BROADCAST_TO(int32);
REGISTER_BROADCAST_TO(int64);

#undef REGISTER_BROADCAST_TO

}  // namespace musa
}  // namespace tensorflow
