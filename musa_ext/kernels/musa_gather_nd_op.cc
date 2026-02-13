#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherNdOp : public MusaOpKernel {
 public:
  explicit MusaGatherNdOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    const int64_t params_dims = params.dims();
    const int64_t indices_dims = indices.dims();
    const int64_t index_depth = indices.dim_size(indices_dims - 1);

    OP_REQUIRES(ctx, index_depth <= params_dims,
                errors::InvalidArgument("index_depth (", index_depth,
                                        ") must be <= params_dims (",
                                        params_dims, ")"));

    TensorShape output_shape;
    for (int i = 0; i < indices_dims - 1; ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int i = index_depth; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    mTensor t_params = CreateMTensor(params);
    mTensor t_indices = CreateMTensor(indices);
    mTensor t_output = CreateMTensor(*output);

    mGatherX op;

    auto status = op.SetMode(::musa::dnn::GatherX::Mode::GATHER_ND);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN GatherX SetMode(GATHER_ND) failed"));

    status = op.Run(handle, t_output, t_indices, t_params);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal(
                    "MUSA muDNN GatherNd (GatherX) execution failed. Status: ",
                    static_cast<int>(status)));
  }
};

#define REGISTER_MUSA_GATHER_ND(type, itype)                      \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                        \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<itype>("Tindices"), \
                          MusaGatherNdOp<type, itype>);

REGISTER_MUSA_GATHER_ND(float, int32);
REGISTER_MUSA_GATHER_ND(float, int64);

REGISTER_MUSA_GATHER_ND(Eigen::half, int32);
REGISTER_MUSA_GATHER_ND(Eigen::half, int64);
REGISTER_MUSA_GATHER_ND(bfloat16, int32);
REGISTER_MUSA_GATHER_ND(bfloat16, int64);

REGISTER_MUSA_GATHER_ND(int32, int32);
REGISTER_MUSA_GATHER_ND(int32, int64);
REGISTER_MUSA_GATHER_ND(int64, int32);
REGISTER_MUSA_GATHER_ND(int64, int64);

}  // namespace musa
}  // namespace tensorflow
