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

// MusaNormalize op: Fused normalization with clip(sqrt(var), eps) pattern
//
// This op implements the following computation:
//   mean = reduce_mean(x, axis)
//   variance = reduce_mean((x - mean)^2, axis)
//   clipped_std = max(sqrt(variance), epsilon)
//   output = (x - mean) / clipped_std
//
// Mathematical equivalence with LayerNorm:
//   LayerNorm: output = (x - mean) / sqrt(variance + eps)
//   NormalizeFusion: output = (x - mean) / max(sqrt(variance), eps)
//
// Note: These are approximately equivalent for most use cases:
//   When var >> eps^2: sqrt(var + eps^2) ≈ sqrt(var)
//   When var << eps^2: sqrt(var + eps^2) ≈ eps

#include <limits>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// Declaration of kernel launcher
template <typename T>
void LaunchNormalize(const T* src, T* dst, int64_t num_rows, int64_t row_size,
                     float epsilon, float max_std, musaStream_t stream);

template <typename T>
class MusaNormalizeOp : public MusaOpKernel {
 public:
  explicit MusaNormalizeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    // max_std defaults to infinity (no upper limit) if not specified
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_std", &max_std_));
    if (max_std_ <= 0.0f) {
      max_std_ = std::numeric_limits<float>::max();
    }
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, "MusaNormalize");

    const Tensor& x = ctx->input(0);
    // gamma and beta are inputs 1 and 2, but we ignore them
    // We always use gamma=1.0 and beta=0.0 for pure normalization

    OP_REQUIRES(ctx, x.dims() >= 1,
                errors::InvalidArgument("Input rank must be >= 1"));

    const int64_t last_dim = x.dim_size(x.dims() - 1);
    const int64_t num_rows = x.NumElements() / last_dim;

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    if (y->NumElements() == 0) {
      VLOG(1) << "MusaNormalizeOp::Compute skipped empty tensor";
      return;
    }

    const T* input_ptr = x.flat<T>().data();
    T* output_ptr = y->flat<T>().data();

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    VLOG(1) << "MusaNormalizeOp::Compute launching kernel, num_rows="
            << num_rows << ", row_size=" << last_dim << ", epsilon=" << epsilon_
            << ", max_std=" << max_std_;

    MUSA_KERNEL_TRACE_START("Kernel");
    LaunchNormalize<T>(input_ptr, output_ptr, num_rows, last_dim, epsilon_,
                       max_std_, stream);
    MUSA_KERNEL_TRACE_END("Kernel");

    VLOG(1) << "MusaNormalizeOp::Compute finished";
  }

 private:
  float epsilon_;
  float max_std_;
};

// Register MUSA kernel
#define REGISTER_MUSA_NORMALIZE(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MusaNormalize").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaNormalizeOp<TYPE>);

REGISTER_MUSA_NORMALIZE(float);
REGISTER_MUSA_NORMALIZE(Eigen::half);
REGISTER_MUSA_NORMALIZE(bfloat16);

#undef REGISTER_MUSA_NORMALIZE

}  // namespace musa

REGISTER_OP("MusaNormalize")
    .Input("x: T")
    .Input("gamma: T")  // Ignored, always use 1.0
    .Input("beta: T")   // Ignored, always use 0.0
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 1e-11")
    .Attr("max_std: float = inf")  // Maximum standard deviation limit
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
