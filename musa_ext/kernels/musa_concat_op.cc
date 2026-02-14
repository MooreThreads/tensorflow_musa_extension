#include <musa_runtime.h>

#include <vector>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
class MusaConcatOp : public MusaOpKernel {
 public:
  explicit MusaConcatOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const int N = ctx->num_inputs() - 1;
    const Tensor& axis_tensor = ctx->input(N);
    int64 axis_val = axis_tensor.scalar<Tidx>()();

    int64_t total_elements = 0;
    int first_non_empty_idx = -1;
    std::vector<int> non_empty_indices;

    for (int i = 0; i < N; ++i) {
      const Tensor& t = ctx->input(i);
      if (t.NumElements() > 0) {
        total_elements += t.NumElements();
        non_empty_indices.push_back(i);
        if (first_non_empty_idx == -1) first_non_empty_idx = i;
      }
    }

    const Tensor& ref =
        ctx->input(first_non_empty_idx == -1 ? 0 : first_non_empty_idx);
    const int dims = ref.dims();
    int normalized_axis = axis_val < 0 ? axis_val + dims : axis_val;

    TensorShape out_shape = ref.shape();
    int64 concat_dim_total = 0;
    for (int i = 0; i < N; ++i) {
      concat_dim_total += ctx->input(i).dim_size(normalized_axis);
    }
    out_shape.set_dim(normalized_axis, concat_dim_total);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (total_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    if (non_empty_indices.size() == 1) {
      const Tensor& src = ctx->input(non_empty_indices[0]);
      musaMemcpyAsync(const_cast<char*>(output->tensor_data().data()),
                      src.tensor_data().data(), src.TotalBytes(),
                      musaMemcpyDeviceToDevice, stream);
      return;
    }

    std::vector<::musa::dnn::Tensor> mudnn_ins;
    mudnn_ins.reserve(non_empty_indices.size());
    for (int idx : non_empty_indices) {
      mudnn_ins.push_back(CreateMTensor(ctx->input(idx), format_));
    }

    ::musa::dnn::Tensor mudnn_out = CreateMTensor(*output, format_);
    ::musa::dnn::Concat concat_op;
    concat_op.SetAxis(normalized_axis);

    auto status =
        concat_op.Run(handle, mudnn_out, static_cast<int>(mudnn_ins.size()),
                      mudnn_ins.data());

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Concat Run failed. Status: ", (int)status));
  }
};

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device("MUSA")
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<float, int32>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device("MUSA")
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int64>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<float, int64>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device("MUSA")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<int32, int32>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device("MUSA")
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<int64, int32>);

}  // namespace musa
}  // namespace tensorflow
