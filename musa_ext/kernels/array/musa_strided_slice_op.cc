#include <mudnn.h>

#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T>
inline bool NeedsHostVisibleSliceSync() {
  return std::is_same<T, int32>::value || std::is_same<T, int64>::value;
}

inline void SyncSliceStreamIfNeeded(OpKernelContext* context, mHandle& handle,
                                    bool should_sync) {
  if (!should_sync) return;

  musaError_t err = musaStreamSynchronize(
      reinterpret_cast<musaStream_t>(handle.GetStream()));
  OP_REQUIRES(context, err == musaSuccess,
              errors::Internal("MUSA StridedSlice stream sync failed: ",
                               musaGetErrorString(err)));
}

inline int64_t ComputeSliceLength(int64_t begin, int64_t end, int64_t stride) {
  int64_t size = 0;
  for (int64_t idx = begin; stride > 0 ? idx < end : idx > end; idx += stride) {
    ++size;
  }
  return size;
}

template <typename T>
void ComputeHostStridedSliceInt32(
    OpKernelContext* context, const Tensor& input,
    const PartialTensorShape& processing_shape, const PartialTensorShape& final_shape,
    bool is_identity, const gtl::InlinedVector<int64_t, 4>& begin,
    const gtl::InlinedVector<int64_t, 4>& end,
    const gtl::InlinedVector<int64_t, 4>& strides) {
  TensorShape final_tensor_shape;
  final_shape.AsTensorShape(&final_tensor_shape);

  if (is_identity) {
    Tensor tmp;
    OP_REQUIRES(context, tmp.CopyFrom(input, final_tensor_shape),
                errors::Internal("MUSA StridedSlice host CopyFrom failed."));
    context->set_output(0, tmp);
    return;
  }

  Tensor* result = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, final_tensor_shape, &result));

  if (result->NumElements() == 0 || input.NumElements() == 0) return;

  TensorShape processing_tensor_shape;
  processing_shape.AsTensorShape(&processing_tensor_shape);
  const int processing_dims = processing_tensor_shape.dims();

  std::vector<int64_t> input_dims(processing_dims, 1);
  std::vector<int64_t> input_strides(processing_dims, 1);
  std::vector<int64_t> slice_sizes(processing_dims, 1);
  for (int i = 0; i < processing_dims; ++i) {
    input_dims[i] = processing_tensor_shape.dim_size(i);
  }
  for (int i = processing_dims - 2; i >= 0; --i) {
    input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
  }
  for (int i = 0; i < processing_dims; ++i) {
    slice_sizes[i] = ComputeSliceLength(begin[i], end[i], strides[i]);
  }

  int64_t expected_elements = 1;
  for (int64_t size : slice_sizes) expected_elements *= size;
  OP_REQUIRES(
      context, expected_elements == result->NumElements(),
      errors::Internal("MUSA host StridedSlice shape mismatch: expected ",
                       expected_elements, " elements from validated slice, got ",
                       result->NumElements()));

  auto input_flat = input.flat<T>();
  auto output_flat = result->flat<T>();
  const T* input_ptr = input_flat.data();
  T* output_ptr = output_flat.data();

  if (processing_dims == 0) {
    output_ptr[0] = input_ptr[0];
    return;
  }

  std::vector<int64_t> output_index(processing_dims, 0);
  int64_t written = 0;
  while (true) {
    int64_t input_offset = 0;
    for (int i = 0; i < processing_dims; ++i) {
      input_offset += (begin[i] + output_index[i] * strides[i]) * input_strides[i];
    }
    output_ptr[written++] = input_ptr[input_offset];

    int dim = processing_dims - 1;
    while (dim >= 0) {
      ++output_index[dim];
      if (output_index[dim] < slice_sizes[dim]) break;
      output_index[dim] = 0;
      --dim;
    }
    if (dim < 0) break;
  }
}

template <typename T>
bool MaybeComputeHostStridedSlice(OpKernelContext* context, const Tensor& input,
                                  const PartialTensorShape& processing_shape,
                                  const PartialTensorShape& final_shape,
                                  bool is_identity,
                                  const gtl::InlinedVector<int64_t, 4>& begin,
                                  const gtl::InlinedVector<int64_t, 4>& end,
                                  const gtl::InlinedVector<int64_t, 4>& strides) {
  return false;
}

template <>
bool MaybeComputeHostStridedSlice<int32>(
    OpKernelContext* context, const Tensor& input,
    const PartialTensorShape& processing_shape,
    const PartialTensorShape& final_shape, bool is_identity,
    const gtl::InlinedVector<int64_t, 4>& begin,
    const gtl::InlinedVector<int64_t, 4>& end,
    const gtl::InlinedVector<int64_t, 4>& strides) {
  ComputeHostStridedSliceInt32<int32>(context, input, processing_shape,
                                      final_shape, is_identity, begin, end,
                                      strides);
  return true;
}

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

  // StridedSlice is memory-intensive but not computationally expensive
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    MUSA_DEBUG_LOG_KERNEL(context);
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();

    PartialTensorShape processing_shape, final_shape;
    bool is_identity = true, slice_dim0 = true, is_simple_slice = true;
    gtl::InlinedVector<int64_t, 4> begin, end, strides;

    OP_REQUIRES_OK(
        context, ::tensorflow::ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     input.shape(), begin_mask_, end_mask_, ellipsis_mask_,
                     new_axis_mask_, shrink_axis_mask_, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    if (MaybeComputeHostStridedSlice<T>(context, input, processing_shape,
                                        final_shape, is_identity, begin, end,
                                        strides)) {
      return;
    }

    if (is_identity) {
      TensorShape final_tensor_shape;
      final_shape.AsTensorShape(&final_tensor_shape);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, final_tensor_shape, &output));
      if (input.NumElements() > 0) {
        auto& h = GetHandleByCtx(context);
        musaError_t err = musaMemcpyAsync(
            output->flat<T>().data(), input.flat<T>().data(),
            input.TotalBytes(), musaMemcpyDeviceToDevice,
            reinterpret_cast<musaStream_t>(h.GetStream()));
        OP_REQUIRES(context, err == musaSuccess,
                    errors::Internal("MUSA StridedSlice memcpy failed: ",
                                     musaGetErrorString(err)));
        SyncSliceStreamIfNeeded(context, h, NeedsHostVisibleSliceSync<T>());
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
    SyncSliceStreamIfNeeded(context, h, NeedsHostVisibleSliceSync<T>());
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
REGISTER_STRIDED_SLICE_MUSA(int64);
REGISTER_STRIDED_SLICE_MUSA(Eigen::half);
REGISTER_STRIDED_SLICE_MUSA(Eigen::bfloat16);
REGISTER_STRIDED_SLICE_MUSA(bool);

REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device("MUSA")
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        MusaStridedSliceOp<int32>);

#undef REGISTER_STRIDED_SLICE_MUSA

}  // namespace
}  // namespace musa
}  // namespace tensorflow
