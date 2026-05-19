#include <musa_runtime.h>

#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

extern "C" {
void LaunchConcatWithSliceGradFloat(const float* slice_grad,
                                    const float* input1, const float* input2,
                                    float* output, int outer, int axis_dim,
                                    int slice_start, int inner_dim,
                                    musaStream_t stream);
void LaunchConcatWithSliceGradBFloat16(const void* slice_grad,
                                       const void* input1, const void* input2,
                                       void* output, int outer, int axis_dim,
                                       int slice_start, int inner_dim,
                                       musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T, typename Tidx>
class MusaConcatOp : public MusaOpKernel {
 public:
  explicit MusaConcatOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Concat is memory-intensive but not computationally expensive
  bool IsExpensive() override { return false; }

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
    if (non_empty_indices.size() == 1) {
      const std::vector<int> forwardable_input_indices = {non_empty_indices[0]};
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              forwardable_input_indices, 0, out_shape,
                              &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    }

    if (total_elements == 0) return;

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("MUSA stream is unavailable for Concat"));

    if (non_empty_indices.size() == 1) {
      const Tensor& src = ctx->input(non_empty_indices[0]);
      if (output->tensor_data().data() == src.tensor_data().data()) {
        return;
      }
      musaError_t err =
          musaMemcpyAsync(const_cast<char*>(output->tensor_data().data()),
                          src.tensor_data().data(), src.TotalBytes(),
                          musaMemcpyDeviceToDevice, stream);
      if (err != musaSuccess) {
        ctx->CtxFailure(errors::Internal("musaMemcpyAsync failed: ",
                                         musaGetErrorString(err)));
        return;
      }
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

class MusaConcatWithSliceGradOp : public MusaOpKernel {
 public:
  explicit MusaConcatWithSliceGradOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis_dim", &axis_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_start", &slice_start_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("inner_dim", &inner_dim_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == 3,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad expected 3 inputs, got ",
                    ctx->num_inputs()));
    OP_REQUIRES(ctx, axis_dim_ > 0 && inner_dim_ > 0 &&
                         slice_start_ >= 0 && slice_start_ < axis_dim_,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad invalid slice attrs"));

    const Tensor& slice_grad = ctx->input(0);
    const Tensor& input1 = ctx->input(1);
    const Tensor& input2 = ctx->input(2);
    const TensorShape& base_shape = input1.shape();
    OP_REQUIRES(ctx, base_shape.dims() == 3,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad requires rank-3 full inputs"));
    OP_REQUIRES(ctx, input2.shape() == base_shape,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad full input shape mismatch: ",
                    input2.shape().DebugString(), " vs ",
                    base_shape.DebugString()));
    OP_REQUIRES(ctx, base_shape.dim_size(1) == axis_dim_ &&
                         base_shape.dim_size(2) == inner_dim_,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad full input dimensions mismatch"));

    const int64_t outer = base_shape.dim_size(0);
    const int64_t expected_slice_elements =
        outer * (axis_dim_ - slice_start_) * inner_dim_;
    OP_REQUIRES(ctx, slice_grad.NumElements() == expected_slice_elements,
                errors::InvalidArgument(
                    "MusaConcatWithSliceGrad slice_grad elements mismatch: ",
                    slice_grad.NumElements(), " vs ",
                    expected_slice_elements));

    TensorShape out_shape = base_shape;
    out_shape.set_dim(2, static_cast<int64_t>(inner_dim_) * 3);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
    if (output->dtype() == DT_FLOAT) {
      LaunchConcatWithSliceGradFloat(
          slice_grad.flat<float>().data(), input1.flat<float>().data(),
          input2.flat<float>().data(), output->flat<float>().data(),
          static_cast<int>(outer), axis_dim_, slice_start_, inner_dim_, stream);
    } else {
      LaunchConcatWithSliceGradBFloat16(
          slice_grad.flat<bfloat16>().data(), input1.flat<bfloat16>().data(),
          input2.flat<bfloat16>().data(), output->flat<bfloat16>().data(),
          static_cast<int>(outer), axis_dim_, slice_start_, inner_dim_, stream);
    }
    musaError_t err = musaGetLastError();
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MusaConcatWithSliceGrad launch failed: ",
                                 musaGetErrorString(err)));
  }

 private:
  int axis_dim_ = 0;
  int slice_start_ = 0;
  int inner_dim_ = 0;
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
                            .TypeConstraint<bfloat16>("T")
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<bfloat16, int32>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("T")
                            .TypeConstraint<int64>("Tidx")
                            .HostMemory("axis"),
                        MusaConcatOp<bfloat16, int64>);

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

REGISTER_KERNEL_BUILDER(Name("MusaConcatWithSliceGrad")
                            .Device("MUSA")
                            .TypeConstraint<float>("T"),
                        MusaConcatWithSliceGradOp);
REGISTER_KERNEL_BUILDER(Name("MusaConcatWithSliceGrad")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("T"),
                        MusaConcatWithSliceGradOp);

}  // namespace musa

REGISTER_OP("MusaConcatWithSliceGrad")
    .Input("slice_grad: T")
    .Input("input1: T")
    .Input("input2: T")
    .Output("output: T")
    .Attr("T: {float, bfloat16}")
    .Attr("axis_dim: int")
    .Attr("slice_start: int")
    .Attr("inner_dim: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return OkStatus();
    });

}  // namespace tensorflow
