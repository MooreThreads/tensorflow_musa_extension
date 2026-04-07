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
// - batch_dims is an explicit ATTRIBUTE (defaults to 0), not inferred from shapes
// - When batch_dims > 0, the first batch_dims dimensions of indices and params
//   are treated as batch dimensions (must match in size)
// - axis must be >= batch_dims
// - Output shape: params.shape[:batch_dims] + indices.shape[batch_dims:] + params.shape[axis+1:]
//
// MuDNN GatherX API:
// - SetAxis(axis): the axis in params to gather from (relative to full params shape)
// - SetBatchDims(batch_dims): number of leading batch dimensions in indices
// - Run(handle, output, indices, params): execute the gather operation

#include <mudnn.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherV2Op : public MusaOpKernel {
 public:
  explicit MusaGatherV2Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    // Read batch_dims as an explicit attribute from the op definition
    // This is the correct way to get batch_dims - it should NOT be inferred from shapes
    // TensorFlow GatherV2 op passes batch_dims as an attribute, not as input
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_dims", &batch_dims_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& axis_tensor = ctx->input(2);

    // Validate axis
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be a scalar, got shape: ",
                                        axis_tensor.shape().DebugString()));

    int64_t axis = 0;
    if (axis_tensor.dtype() == DT_INT32) {
      axis = static_cast<int64_t>(axis_tensor.scalar<int32>()());
    } else if (axis_tensor.dtype() == DT_INT64) {
      axis = axis_tensor.scalar<int64>()();
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("axis must be int32 or int64, got: ",
                                          DataTypeString(axis_tensor.dtype())));
    }

    const int64_t params_dims = params.dims();
    const int indices_dims = indices.dims();

    // Normalize negative axis
    if (axis < 0) {
      axis += params_dims;
    }

    // Normalize negative batch_dims (relative to indices rank)
    // TensorFlow supports batch_dims=-1 to mean indices.dims()-1
    int batch_dims = batch_dims_;
    if (batch_dims < 0) {
      batch_dims += indices_dims;
    }

    OP_REQUIRES(ctx, axis >= 0 && axis < params_dims,
                errors::InvalidArgument("Expected axis in range [", -params_dims,
                                        ", ", params_dims, "), but got ", axis));

    // batch_dims must be in range [0, min(axis, indices_dims)]
    OP_REQUIRES(ctx, batch_dims >= 0 && batch_dims <= std::min(axis, static_cast<int64_t>(indices_dims)),
                errors::InvalidArgument("batch_dims must be in range [0, min(axis, indices.dims())], "
                                        "got batch_dims=", batch_dims, ", axis=", axis,
                                        ", indices.dims()=", indices_dims));

    // Validate indices dtype
    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64, got: ",
                                        DataTypeString(indices.dtype())));

    // Validate batch dimension sizes match between params and indices
    // When batch_dims > 0, params.shape[:batch_dims] must equal indices.shape[:batch_dims]
    for (int i = 0; i < batch_dims; ++i) {
      OP_REQUIRES(ctx, params.dim_size(i) == indices.dim_size(i),
                  errors::InvalidArgument("batch dimension ", i, " must match: "
                                          "params.dim_size(", i, ")=", params.dim_size(i),
                                          " != indices.dim_size(", i, ")=", indices.dim_size(i)));
    }

    // Build output shape according to TensorFlow GatherV2 specification:
    // output_shape = params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis+1:]
    //
    // Note: The formula is params.shape[:AXIS] not params.shape[:batch_dims]
    // This is the key difference that was causing the bug.
    // When batch_dims > 0, the batch dimensions from params[:batch_dims] are included
    // in params[:axis] since axis >= batch_dims is required.
    TensorShape output_shape;

    // Add all params dimensions before axis (including batch dimensions)
    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    // Add indices dimensions after batch_dims (the index dimensions)
    for (int64_t i = batch_dims; i < indices_dims; ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }

    // Add params dimensions after axis (the remaining params dims)
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) {
      return;
    }

    // Use muDNN GatherX for the operation
    auto& handle = GetHandleByCtx(ctx);

    mTensor params_mt = CreateMTensor(params, format_);
    mTensor indices_mt = CreateMTensor(indices, format_);
    mTensor output_mt = CreateMTensor(*output, format_);

    ::musa::dnn::GatherX gather_op;
    gather_op.SetMode(::musa::dnn::GatherX::Mode::GATHER);
    // Note: SetAxis takes the axis relative to params shape
    // MuDNN GatherX internally handles batch_dims by treating first batch_dims dims as batch
    gather_op.SetAxis(static_cast<int>(axis));
    gather_op.SetBatchDims(batch_dims);

    // Run the gather operation
    // GatherX::Run signature: (handle, output, indices, params)
    auto status = gather_op.Run(handle, output_mt, indices_mt, params_mt);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("GatherX execution failed. Status: ",
                                 static_cast<int>(status)));
  }

 private:
  int batch_dims_ = 0;  // Explicit attribute from op definition
};

// Registration macros
#define REGISTER_GATHER_V2_MUDNN(T, IndexT)                        \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                         \
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
#define REGISTER_GATHER_V1_MUDNN(T, IndexT)      \
  REGISTER_KERNEL_BUILDER(Name("Gather")        \
                              .Device(DEVICE_MTGPU) \
                              .TypeConstraint<T>("Tparams") \
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