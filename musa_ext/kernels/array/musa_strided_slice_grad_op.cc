#include <musa_runtime.h>

#include <cstdint>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace musa {
namespace {

constexpr int kMaxStridedSliceGradDims = 8;

struct StridedSliceGradLaunchParams {
  int rank;
  int64_t processing_shape[kMaxStridedSliceGradDims];
  int64_t output_strides[kMaxStridedSliceGradDims];
  int64_t begin[kMaxStridedSliceGradDims];
  int64_t strides[kMaxStridedSliceGradDims];
};

extern "C" {
void LaunchStridedSliceGradFloat(const float* dy, float* output,
                                 int64_t total_elements,
                                 StridedSliceGradLaunchParams params,
                                 musaStream_t stream);
void LaunchStridedSliceGradDouble(const double* dy, double* output,
                                  int64_t total_elements,
                                  StridedSliceGradLaunchParams params,
                                  musaStream_t stream);
void LaunchStridedSliceGradInt32(const int32* dy, int32* output,
                                 int64_t total_elements,
                                 StridedSliceGradLaunchParams params,
                                 musaStream_t stream);
void LaunchStridedSliceGradInt64(const int64_t* dy, int64_t* output,
                                 int64_t total_elements,
                                 StridedSliceGradLaunchParams params,
                                 musaStream_t stream);
void LaunchStridedSliceGradBool(const bool* dy, bool* output,
                                int64_t total_elements,
                                StridedSliceGradLaunchParams params,
                                musaStream_t stream);
void LaunchStridedSliceGradHalf(const void* dy, void* output,
                                int64_t total_elements,
                                StridedSliceGradLaunchParams params,
                                musaStream_t stream);
void LaunchStridedSliceGradBFloat16(const void* dy, void* output,
                                    int64_t total_elements,
                                    StridedSliceGradLaunchParams params,
                                    musaStream_t stream);
}

template <typename Index>
Status ShapeTensorToTensorShape(const Tensor& shape_tensor,
                                TensorShape* output_shape) {
  if (!TensorShapeUtils::IsVector(shape_tensor.shape())) {
    return errors::InvalidArgument("shape must be a vector, got ",
                                   shape_tensor.shape().DebugString());
  }

  auto flat = shape_tensor.flat<Index>();
  for (int i = 0; i < flat.size(); ++i) {
    const int64_t dim = static_cast<int64_t>(flat(i));
    if (dim < 0) {
      return errors::InvalidArgument("shape contains negative dimension ", dim,
                                     " at index ", i);
    }
    output_shape->AddDim(dim);
  }
  return Status::OK();
}

inline void FillLaunchParams(const TensorShape& output_shape,
                             const TensorShape& processing_shape,
                             const gtl::InlinedVector<int64_t, 4>& begin,
                             const gtl::InlinedVector<int64_t, 4>& strides,
                             StridedSliceGradLaunchParams* params) {
  params->rank = output_shape.dims();

  int64_t stride = 1;
  for (int dim = output_shape.dims() - 1; dim >= 0; --dim) {
    params->output_strides[dim] = stride;
    stride *= output_shape.dim_size(dim);
  }

  for (int dim = 0; dim < output_shape.dims(); ++dim) {
    params->processing_shape[dim] = processing_shape.dim_size(dim);
    params->begin[dim] = begin[dim];
    params->strides[dim] = strides[dim];
  }
}

template <typename T>
struct StridedSliceGradLauncher;

template <>
struct StridedSliceGradLauncher<float> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradFloat(dy.flat<float>().data(),
                                output->flat<float>().data(), total_elements,
                                params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<double> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradDouble(dy.flat<double>().data(),
                                 output->flat<double>().data(), total_elements,
                                 params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<int32> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradInt32(dy.flat<int32>().data(),
                                output->flat<int32>().data(), total_elements,
                                params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<int64_t> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradInt64(dy.flat<int64_t>().data(),
                                output->flat<int64_t>().data(), total_elements,
                                params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<bool> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradBool(dy.flat<bool>().data(),
                               output->flat<bool>().data(), total_elements,
                               params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<Eigen::half> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradHalf(dy.data(), output->data(), total_elements,
                               params, stream);
  }
};

template <>
struct StridedSliceGradLauncher<Eigen::bfloat16> {
  static void Run(const Tensor& dy, Tensor* output, int64_t total_elements,
                  const StridedSliceGradLaunchParams& params,
                  musaStream_t stream) {
    LaunchStridedSliceGradBFloat16(dy.data(), output->data(), total_elements,
                                   params, stream);
  }
};

template <typename T, typename Index>
class MusaStridedSliceGradOp : public OpKernel {
 public:
  explicit MusaStridedSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask_));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    const Tensor& shape_tensor = context->input(0);
    const Tensor& dy = context->input(4);

    TensorShape output_shape;
    OP_REQUIRES_OK(
        context, ShapeTensorToTensorShape<Index>(shape_tensor, &output_shape));

    OP_REQUIRES(context, output_shape.dims() <= kMaxStridedSliceGradDims,
                errors::Unimplemented("MUSA StridedSliceGrad supports rank <= ",
                                      kMaxStridedSliceGradDims, ", got ",
                                      output_shape.dims()));

    TensorShape processing_shape;
    TensorShape final_shape;
    bool is_identity = true;
    bool is_simple_slice = true;
    bool slice_dim0 = true;
    gtl::InlinedVector<int64_t, 4> begin, end, strides;

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     output_shape, begin_mask_, end_mask_, ellipsis_mask_,
                     new_axis_mask_, shrink_axis_mask_, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    OP_REQUIRES(context, processing_shape.num_elements() == dy.NumElements(),
                errors::InvalidArgument(
                    "dy has ", dy.NumElements(),
                    " elements, but StridedSliceGrad processing shape ",
                    processing_shape.DebugString(), " has ",
                    processing_shape.num_elements(), " elements"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    mHandle& handle = GetHandleByCtx(context);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    musaError_t err =
        musaMemsetAsync(output->data(), 0, output->TotalBytes(), stream);
    OP_REQUIRES(context, err == musaSuccess,
                errors::Internal("MUSA StridedSliceGrad memset failed: ",
                                 musaGetErrorString(err)));

    if (dy.NumElements() == 0) return;

    StridedSliceGradLaunchParams params = {};
    FillLaunchParams(output_shape, processing_shape, begin, strides, &params);
    StridedSliceGradLauncher<T>::Run(dy, output, dy.NumElements(), params,
                                     stream);

    err = musaGetLastError();
    OP_REQUIRES(context, err == musaSuccess,
                errors::Internal("MUSA StridedSliceGrad launch failed: ",
                                 musaGetErrorString(err)));
  }

 private:
  int32 begin_mask_, end_mask_;
  int32 ellipsis_mask_, new_axis_mask_, shrink_axis_mask_;
};

#define REGISTER_STRIDED_SLICE_GRAD_MUSA(T, Index)            \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")            \
                              .Device("MUSA")                 \
                              .TypeConstraint<T>("T")         \
                              .TypeConstraint<Index>("Index") \
                              .HostMemory("shape")            \
                              .HostMemory("begin")            \
                              .HostMemory("end")              \
                              .HostMemory("strides"),         \
                          MusaStridedSliceGradOp<T, Index>)

#define REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(T) \
  REGISTER_STRIDED_SLICE_GRAD_MUSA(T, int32);       \
  REGISTER_STRIDED_SLICE_GRAD_MUSA(T, int64_t)

REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(float);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(double);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(int32);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(int64_t);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(Eigen::half);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(Eigen::bfloat16);
REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES(bool);

#undef REGISTER_STRIDED_SLICE_GRAD_MUSA_INDICES
#undef REGISTER_STRIDED_SLICE_GRAD_MUSA

}  // namespace
}  // namespace musa
}  // namespace tensorflow
