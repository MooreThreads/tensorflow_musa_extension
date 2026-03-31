/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// GatherV2 op implementation using muDNN GatherX for proper batch_dims support.
//
// TensorFlow GatherV2 op specification:
// - batch_dims is an explicit ATTRIBUTE (defaults to 0), not inferred from
// shapes
// - When batch_dims > 0, the first batch_dims dimensions of indices and params
//   are treated as batch dimensions (must match in size)
// - axis must be >= batch_dims
// - Output shape: params.shape[:batch_dims] + indices.shape[batch_dims:] +
// params.shape[axis+1:]
//
// MuDNN GatherX API:
// - SetAxis(axis): the axis in params to gather from (relative to full params
// shape)
// - SetBatchDims(batch_dims): number of leading batch dimensions in indices
// - Run(handle, output, indices, params): execute the gather operation

#include <mudnn.h>

#include <algorithm>

#include "../utils_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

// Custom kernel launchers (provide GPU-side index clamping)
extern "C" {
void LaunchGatherV2FloatInt32(const float* params, const int* indices,
                              float* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int limit, musaStream_t stream);
void LaunchGatherV2FloatInt64(const float* params, const int64_t* indices,
                              float* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int64_t limit, musaStream_t stream);
void LaunchGatherV2DoubleInt32(const double* params, const int* indices,
                               double* output, int64_t batch_size,
                               int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride,
                               int limit, musaStream_t stream);
void LaunchGatherV2DoubleInt64(const double* params, const int64_t* indices,
                               double* output, int64_t batch_size,
                               int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride,
                               int64_t limit, musaStream_t stream);
void LaunchGatherV2Int32Int32(const int* params, const int* indices,
                              int* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int limit, musaStream_t stream);
void LaunchGatherV2Int32Int64(const int* params, const int64_t* indices,
                              int* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int64_t limit, musaStream_t stream);
void LaunchGatherV2Int64Int32(const int64_t* params, const int* indices,
                              int64_t* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int limit, musaStream_t stream);
void LaunchGatherV2Int64Int64(const int64_t* params, const int64_t* indices,
                              int64_t* output, int64_t batch_size,
                              int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride,
                              int64_t limit, musaStream_t stream);
void LaunchGatherV2BoolInt32(const bool* params, const int* indices,
                             bool* output, int64_t batch_size,
                             int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride,
                             int limit, musaStream_t stream);
void LaunchGatherV2BoolInt64(const bool* params, const int64_t* indices,
                             bool* output, int64_t batch_size,
                             int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride,
                             int64_t limit, musaStream_t stream);
void LaunchGatherV2HalfInt32(const void* params, const int* indices,
                             void* output, int64_t batch_size,
                             int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride,
                             int limit, musaStream_t stream);
void LaunchGatherV2HalfInt64(const void* params, const int64_t* indices,
                             void* output, int64_t batch_size,
                             int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride,
                             int64_t limit, musaStream_t stream);
void LaunchGatherV2BFloat16Int32(const void* params, const int* indices,
                                 void* output, int64_t batch_size,
                                 int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride,
                                 int limit, musaStream_t stream);
void LaunchGatherV2BFloat16Int64(const void* params, const int64_t* indices,
                                 void* output, int64_t batch_size,
                                 int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride,
                                 int64_t limit, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherV2Op : public MusaOpKernel {
 public:
  explicit MusaGatherV2Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_dims", &batch_dims_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& axis_tensor = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be a scalar, got shape: ",
                                        axis_tensor.shape().DebugString()));

    int64_t axis = 0;
    if (axis_tensor.dtype() == DT_INT32) {
      axis = axis_tensor.scalar<int32>()();
    } else if (axis_tensor.dtype() == DT_INT64) {
      axis = axis_tensor.scalar<int64>()();
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("axis must be int32 or int64, got: ",
                                          DataTypeString(axis_tensor.dtype())));
    }

    const int64_t params_dims = params.dims();
    const int indices_dims = indices.dims();
    if (axis < 0) {
      axis += params_dims;
    }

    int batch_dims = batch_dims_;
    if (batch_dims < 0) {
      batch_dims += indices_dims;
    }

    OP_REQUIRES(
        ctx, axis >= 0 && axis < params_dims,
        errors::InvalidArgument("Expected axis in range [", -params_dims, ", ",
                                params_dims, "), but got ", axis));

    OP_REQUIRES(
        ctx,
        batch_dims >= 0 &&
            batch_dims <= std::min(axis, static_cast<int64_t>(indices_dims)),
        errors::InvalidArgument(
            "batch_dims must be in range [0, min(axis, indices.dims())], "
            "got batch_dims=",
            batch_dims, ", axis=", axis, ", indices.dims()=", indices_dims));

    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64, got: ",
                                        DataTypeString(indices.dtype())));

    for (int i = 0; i < batch_dims; ++i) {
      OP_REQUIRES(ctx, params.dim_size(i) == indices.dim_size(i),
                  errors::InvalidArgument("batch dimension ", i,
                                          " must match: "
                                          "params.dim_size(",
                                          i, ")=", params.dim_size(i),
                                          " != indices.dim_size(", i,
                                          ")=", indices.dim_size(i)));
    }

    TensorShape output_shape;
    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int64_t i = batch_dims; i < indices_dims; ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

    // use kernel when batch_dims == 0
    if (batch_dims == 0) {
      const int64_t limit = params.dim_size(axis);

      int64_t batch_size = 1;
      for (int64_t i = 0; i < axis; ++i) {
        batch_size *= params.dim_size(i);
      }

      int64_t inner_size = 1;
      for (int64_t i = axis + 1; i < params_dims; ++i) {
        inner_size *= params.dim_size(i);
      }

      const int64_t indices_size = indices.NumElements();
      const int64_t params_stride = limit * inner_size;

      musaStream_t stream = GetMusaStreamByCtx(ctx);

      LaunchKernel(params.flat<T>().data(), indices.flat<IndexT>().data(),
                   output->flat<T>().data(), batch_size, limit, inner_size,
                   indices_size, params_stride, static_cast<IndexT>(limit),
                   stream);
      return;
    }

    // use mudnn
    auto& handle = GetHandleByCtx(ctx);

    mTensor params_mt = CreateMTensor(params, format_);
    mTensor indices_mt = CreateMTensor(indices, format_);
    mTensor output_mt = CreateMTensor(*output, format_);

    ::musa::dnn::GatherX gather_op;
    gather_op.SetMode(::musa::dnn::GatherX::Mode::GATHER);
    gather_op.SetAxis(static_cast<int>(axis));
    gather_op.SetBatchDims(batch_dims);

    auto status = gather_op.Run(handle, output_mt, indices_mt, params_mt);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("GatherX execution failed. Status: ",
                                 static_cast<int>(status)));
  }

 private:
  int batch_dims_ = 0;  // Explicit attribute from op definition

  void LaunchKernel(const T* params, const IndexT* indices, T* output,
                    int64_t batch_size, int64_t axis_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, IndexT limit,
                    musaStream_t stream);
};

// Launcher specializations
#define DEFINE_GATHER_LAUNCHER(T, IndexT, launcher_func)                      \
  template <>                                                                 \
  void MusaGatherV2Op<T, IndexT>::LaunchKernel(                               \
      const T* params, const IndexT* indices, T* output, int64_t batch_size,  \
      int64_t axis_size, int64_t inner_size, int64_t indices_size,            \
      int64_t params_stride, IndexT limit, musaStream_t stream) {             \
    launcher_func(params, indices, output, batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream);                \
  }

DEFINE_GATHER_LAUNCHER(float, int32, LaunchGatherV2FloatInt32)
DEFINE_GATHER_LAUNCHER(float, int64, LaunchGatherV2FloatInt64)
DEFINE_GATHER_LAUNCHER(double, int32, LaunchGatherV2DoubleInt32)
DEFINE_GATHER_LAUNCHER(double, int64, LaunchGatherV2DoubleInt64)
DEFINE_GATHER_LAUNCHER(int32, int32, LaunchGatherV2Int32Int32)
DEFINE_GATHER_LAUNCHER(int32, int64, LaunchGatherV2Int32Int64)
DEFINE_GATHER_LAUNCHER(int64, int32, LaunchGatherV2Int64Int32)
DEFINE_GATHER_LAUNCHER(int64, int64, LaunchGatherV2Int64Int64)
DEFINE_GATHER_LAUNCHER(bool, int32, LaunchGatherV2BoolInt32)
DEFINE_GATHER_LAUNCHER(bool, int64, LaunchGatherV2BoolInt64)

#define DEFINE_GATHER_LAUNCHER_HALF(IndexT, launcher_func)                   \
  template <>                                                                \
  void MusaGatherV2Op<Eigen::half, IndexT>::LaunchKernel(                    \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size,             \
      int64_t indices_size, int64_t params_stride, IndexT limit,             \
      musaStream_t stream) {                                                 \
    launcher_func(reinterpret_cast<const void*>(params), indices,            \
                  reinterpret_cast<void*>(output), batch_size, axis_size,    \
                  inner_size, indices_size, params_stride, limit, stream);   \
  }

DEFINE_GATHER_LAUNCHER_HALF(int32, LaunchGatherV2HalfInt32)
DEFINE_GATHER_LAUNCHER_HALF(int64, LaunchGatherV2HalfInt64)

#define DEFINE_GATHER_LAUNCHER_BF16(IndexT, launcher_func)                 \
  template <>                                                              \
  void MusaGatherV2Op<bfloat16, IndexT>::LaunchKernel(                     \
      const bfloat16* params, const IndexT* indices, bfloat16* output,     \
      int64_t batch_size, int64_t axis_size, int64_t inner_size,           \
      int64_t indices_size, int64_t params_stride, IndexT limit,           \
      musaStream_t stream) {                                               \
    launcher_func(reinterpret_cast<const void*>(params), indices,          \
                  reinterpret_cast<void*>(output), batch_size, axis_size,  \
                  inner_size, indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER_BF16(int32, LaunchGatherV2BFloat16Int32)
DEFINE_GATHER_LAUNCHER_BF16(int64, LaunchGatherV2BFloat16Int64)

#undef DEFINE_GATHER_LAUNCHER
#undef DEFINE_GATHER_LAUNCHER_HALF
#undef DEFINE_GATHER_LAUNCHER_BF16

// Registration macros
#define REGISTER_GATHER_V2_MUDNN(T, IndexT)                       \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                        \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<IndexT>("Tindices") \
                              .HostMemory("axis"),                \
                          MusaGatherV2Op<T, IndexT>);

// Register for all supported types
REGISTER_GATHER_V2_MUDNN(float, int32);
REGISTER_GATHER_V2_MUDNN(float, int64);
REGISTER_GATHER_V2_MUDNN(double, int32);
REGISTER_GATHER_V2_MUDNN(double, int64);
REGISTER_GATHER_V2_MUDNN(int32, int32);
REGISTER_GATHER_V2_MUDNN(int32, int64);
REGISTER_GATHER_V2_MUDNN(int64, int32);
REGISTER_GATHER_V2_MUDNN(int64, int64);
REGISTER_GATHER_V2_MUDNN(Eigen::half, int32);
REGISTER_GATHER_V2_MUDNN(Eigen::half, int64);
REGISTER_GATHER_V2_MUDNN(bfloat16, int32);
REGISTER_GATHER_V2_MUDNN(bfloat16, int64);

#undef REGISTER_GATHER_V2_MUDNN

// Also register Gather (v1) for backward compatibility
// Note: Gather v1 does NOT have batch_dims attribute, so we use batch_dims=0
#define REGISTER_GATHER_V1_MUDNN(T, IndexT)                        \
  REGISTER_KERNEL_BUILDER(Name("Gather")                           \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<T>("Tparams")        \
                              .TypeConstraint<IndexT>("Tindices"), \
                          MusaGatherV2Op<T, IndexT>);

REGISTER_GATHER_V1_MUDNN(float, int32);
REGISTER_GATHER_V1_MUDNN(float, int64);
REGISTER_GATHER_V1_MUDNN(double, int32);
REGISTER_GATHER_V1_MUDNN(double, int64);
REGISTER_GATHER_V1_MUDNN(int32, int32);
REGISTER_GATHER_V1_MUDNN(int32, int64);
REGISTER_GATHER_V1_MUDNN(int64, int32);
REGISTER_GATHER_V1_MUDNN(int64, int64);
REGISTER_GATHER_V1_MUDNN(Eigen::half, int32);
REGISTER_GATHER_V1_MUDNN(Eigen::half, int64);
REGISTER_GATHER_V1_MUDNN(bfloat16, int32);
REGISTER_GATHER_V1_MUDNN(bfloat16, int64);

#undef REGISTER_GATHER_V1_MUDNN

}  // namespace musa
}  // namespace tensorflow
