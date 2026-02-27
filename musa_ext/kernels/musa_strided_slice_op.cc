#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {
namespace {
template <typename T>
class MusaStridedSliceOp : public OpKernel {
 public:
  explicit MusaStridedSliceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    PartialTensorShape processing_shape, final_shape;
    bool is_identity = true, slice_dim0 = true, is_simple_slice = true;
    gtl::InlinedVector<long long, 4> begin, end, strides;

    OP_REQUIRES_OK(
        context, ::tensorflow::ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     input.shape(), begin_mask_, end_mask_, ellipsis_mask_,
                     new_axis_mask_, shrink_axis_mask_, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    if (is_identity) {
      TensorShape final_tensor_shape;
      final_shape.AsTensorShape(&final_tensor_shape);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, final_tensor_shape, &output));
      if (input.NumElements() > 0) {
        auto& h = GetHandleByCtx(context);
        musaMemcpyAsync(output->flat<T>().data(), input.flat<T>().data(),
                        input.TotalBytes(), musaMemcpyDeviceToDevice,
                        reinterpret_cast<musaStream_t>(h.GetStream()));
      }
      return;
    }

    TensorShape final_tensor_shape;
    final_shape.AsTensorShape(&final_tensor_shape);
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, final_tensor_shape, &result));

    if (result->NumElements() == 0 || input.NumElements() == 0) return;

    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*result);

    TensorShape proc_tensor_shape;
    processing_shape.AsTensorShape(&proc_tensor_shape);
    std::vector<int64_t> out_dims_aligned;
    for (int i = 0; i < proc_tensor_shape.dims(); ++i) {
      out_dims_aligned.push_back(proc_tensor_shape.dim_size(i));
    }
    if (out_dims_aligned.empty()) out_dims_aligned.push_back(1);

    MTOP_CHECK_OK(out_mt.SetNdInfo(static_cast<int>(out_dims_aligned.size()),
                                   out_dims_aligned.data()),
                  "SetNdInfo Out", context);

    std::vector<int64_t> m_starts(input_dims, 0);
    std::vector<int64_t> m_strides(input_dims, 1);

    for (int i = 0; i < input_dims; ++i) {
      if (i < (int)begin.size()) {
        int64_t s_begin = static_cast<int64_t>(begin[i]);
        int64_t s_stride = static_cast<int64_t>(strides[i]);
        int64_t dim_max = input.dim_size(i);

        if (s_stride < 0) {
          if (s_begin >= dim_max) s_begin = dim_max - 1;
          if (s_begin < 0) s_begin += dim_max;
        } else {
          if (s_begin < 0) s_begin = 0;
          if (s_begin >= dim_max && dim_max > 0) s_begin = dim_max - 1;
        }
        m_starts[i] = s_begin;
        m_strides[i] = s_stride;
      }
    }

    mHandle& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;

    MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, m_starts.data(),
                                             m_strides.data()),
                  "ConfigDimStride", context);

    MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "RunOp", context);
    // Note: No explicit sync needed - TF's dependency tracking handles it
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
};

#define REGISTER_STRIDED_SLICE_MUSA(T)                \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")        \
                              .Device("MUSA")         \
                              .TypeConstraint<T>("T") \
                              .HostMemory("begin")    \
                              .HostMemory("end")      \
                              .HostMemory("strides"), \
                          MusaStridedSliceOp<T>)

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
