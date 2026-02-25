#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"
namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherOp : public MusaOpKernel {
 public:
  explicit MusaGatherOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    int64_t axis = 0;
    if (ctx->num_inputs() >= 3) {
      const Tensor& axis_tensor = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be a scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = static_cast<int64_t>(axis_tensor.scalar<int32>()());
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("axis must be int32 or int64"));
      }
    }

    const int64_t params_dims = params.dims();
    if (axis < 0) {
      axis += params_dims;
    }

    OP_REQUIRES(
        ctx, axis >= 0 && axis < params_dims,
        errors::InvalidArgument("Expected axis in the range [", -params_dims,
                                ", ", params_dims, "), but got ", axis));

    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64"));

    TensorShape output_shape;
    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int64_t i = 0; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    const int64_t limit = params.dim_size(axis);
    if (indices.NumElements() > 0) {
      Tensor indices_cpu(indices.dtype(), indices.shape());

      const void* d_ptr =
          static_cast<const void*>(indices.flat<IndexT>().data());
      void* h_ptr = static_cast<void*>(indices_cpu.flat<IndexT>().data());
      size_t bytes = indices.NumElements() * sizeof(IndexT);

      mStatus m_stat = MusaMemcpyD2H(h_ptr, d_ptr, bytes);

      OP_REQUIRES(
          ctx, m_stat == mStatus::SUCCESS,
          errors::Internal("MUSA D2H Memcpy failed for indices check."));

      auto Tindices = indices_cpu.flat<IndexT>();
      for (int64_t i = 0; i < Tindices.size(); ++i) {
        if (Tindices(i) < 0 || Tindices(i) >= limit) {
          OP_REQUIRES(ctx, false,
                      errors::InvalidArgument(
                          "MUSA Gather indices out of range: indices[...] = ",
                          Tindices(i), " is not in [0, ", limit, ")"));
        }
      }
    }

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_params = CreateMTensor(params, format_);
    mTensor t_indices = CreateMTensor(indices, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    mGatherX op;

    OP_REQUIRES(ctx, axis <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("Axis value too large"));
    op.SetAxis(static_cast<int>(axis));

    auto status = op.Run(handle, t_output, t_indices, t_params);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA muDNN Gather execution failed. Status: ",
                                 static_cast<int>(status)));
  }
};

#define REGISTER_GATHER_V2_FULL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V2_FULL(float);
REGISTER_GATHER_V2_FULL(double);
REGISTER_GATHER_V2_FULL(int32);
REGISTER_GATHER_V2_FULL(int64);
REGISTER_GATHER_V2_FULL(bool);
REGISTER_GATHER_V2_FULL(Eigen::half);
REGISTER_GATHER_V2_FULL(bfloat16);

#undef REGISTER_GATHER_V2_FULL

#define REGISTER_GATHER_V1(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaGatherOp<T, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V1(float);
REGISTER_GATHER_V1(double);
REGISTER_GATHER_V1(int32);
REGISTER_GATHER_V1(int64);
REGISTER_GATHER_V1(bool);
REGISTER_GATHER_V1(Eigen::half);
REGISTER_GATHER_V1(bfloat16);

#undef REGISTER_GATHER_V1

}  // namespace musa
}  // namespace tensorflow
