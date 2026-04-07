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

// Implementation of InTopKV2 op for MUSA devices.
// Used in recommendation systems for computing top-k metrics.

#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

// Kernel launcher declared in the .mu file
template <typename T>
void LaunchInTopKV2Int32(const T* predictions, const int32_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream);

template <typename T>
void LaunchInTopKV2Int64(const T* predictions, const int64_t* targets, bool* output,
                         int batch_size, int num_classes, int k,
                         musaStream_t stream);

// Op implementation for int32 targets
class MusaInTopKV2Int32Op : public MusaOpKernel {
 public:
  explicit MusaInTopKV2Int32Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& predictions_t = ctx->input(0);
    const Tensor& targets_t = ctx->input(1);
    const Tensor& k_t = ctx->input(2);

    // Validate inputs
    OP_REQUIRES(ctx, predictions_t.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional, got shape: ",
                                        predictions_t.shape().DebugString()));
    OP_REQUIRES(ctx, targets_t.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional, got shape: ",
                                        targets_t.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(k_t.shape()),
                errors::InvalidArgument("k must be a scalar, got shape: ",
                                        k_t.shape().DebugString()));

    const int64_t batch_size = predictions_t.dim_size(0);
    const int64_t num_classes = predictions_t.dim_size(1);

    OP_REQUIRES(ctx, targets_t.dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "targets must have same batch size as predictions. "
                    "predictions: ", batch_size, ", targets: ", targets_t.dim_size(0)));

    int k = k_t.scalar<int>()();
    OP_REQUIRES(ctx, k >= 0,
                errors::InvalidArgument("k must be >= 0, got: ", k));
    OP_REQUIRES(ctx, k <= num_classes,
                errors::InvalidArgument("k must be <= num_classes (", num_classes,
                                        "), got: ", k));

    // Allocate output
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &output_t));

    // Handle edge cases
    if (batch_size == 0 || num_classes == 0) {
      return;
    }

    // Get device pointers
    const float* predictions_ptr = predictions_t.flat<float>().data();
    const int32_t* targets_ptr = targets_t.flat<int32_t>().data();
    bool* output_ptr = output_t->flat<bool>().data();

    // Launch kernel
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchInTopKV2Int32<float>(predictions_ptr, targets_ptr, output_ptr,
                               batch_size, num_classes, k, stream);

    musaError_t err = musaGetLastError();
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("InTopKV2 kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

// Op implementation for int64 targets
class MusaInTopKV2Int64Op : public MusaOpKernel {
 public:
  explicit MusaInTopKV2Int64Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& predictions_t = ctx->input(0);
    const Tensor& targets_t = ctx->input(1);
    const Tensor& k_t = ctx->input(2);

    // Validate inputs
    OP_REQUIRES(ctx, predictions_t.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional, got shape: ",
                                        predictions_t.shape().DebugString()));
    OP_REQUIRES(ctx, targets_t.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional, got shape: ",
                                        targets_t.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(k_t.shape()),
                errors::InvalidArgument("k must be a scalar, got shape: ",
                                        k_t.shape().DebugString()));

    const int64_t batch_size = predictions_t.dim_size(0);
    const int64_t num_classes = predictions_t.dim_size(1);

    OP_REQUIRES(ctx, targets_t.dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "targets must have same batch size as predictions. "
                    "predictions: ", batch_size, ", targets: ", targets_t.dim_size(0)));

    int64_t k = k_t.scalar<int64_t>()();
    OP_REQUIRES(ctx, k >= 0,
                errors::InvalidArgument("k must be >= 0, got: ", k));
    OP_REQUIRES(ctx, k <= num_classes,
                errors::InvalidArgument("k must be <= num_classes (", num_classes,
                                        "), got: ", k));

    // Allocate output
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &output_t));

    // Handle edge cases
    if (batch_size == 0 || num_classes == 0) {
      return;
    }

    // Get device pointers
    const float* predictions_ptr = predictions_t.flat<float>().data();
    const int64_t* targets_ptr = targets_t.flat<int64_t>().data();
    bool* output_ptr = output_t->flat<bool>().data();

    // Launch kernel
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchInTopKV2Int64<float>(predictions_ptr, targets_ptr, output_ptr,
                               batch_size, num_classes, static_cast<int>(k), stream);

    musaError_t err = musaGetLastError();
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("InTopKV2 kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

// Registration for int32 targets
REGISTER_KERNEL_BUILDER(Name("InTopKV2")
                            .Device(DEVICE_MTGPU)
                            .TypeConstraint<int32>("T"),
                        MusaInTopKV2Int32Op);

// Registration for int64 targets
REGISTER_KERNEL_BUILDER(Name("InTopKV2")
                            .Device(DEVICE_MTGPU)
                            .TypeConstraint<int64>("T"),
                        MusaInTopKV2Int64Op);

}  // namespace musa
}  // namespace tensorflow