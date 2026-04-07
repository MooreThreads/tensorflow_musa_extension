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

/// MusaPRelu fusion op: PRelu(x, alpha) = max(0, x) + alpha * min(0, x)
/// Uses muDNN Binary::Mode::PRELU for efficient computation
#include <mudnn.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

 template <typename T> 
void MusaNegKernelLauncher(const void* in, void* out, int size,
                        musaStream_t stream);
template <typename T>
class MusaPReluOp : public MusaOpKernel {
 public:
  explicit MusaPReluOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // PRelu is element-wise - lightweight
  bool IsExpensive() override { return false; }

  // Forward declaration of neg kernel launcher


  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input = ctx->input(0);   // x
    const Tensor& alpha = ctx->input(1);    // alpha (slope for negative part)

    // Output has the same shape as input
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    // Use output as intermediate storage for -alpha when shapes match
    if (alpha.shape() == input.shape()) {
      // Compute -alpha into output
      MusaNegKernelLauncher<T>(alpha.tensor_data().data(),
                               const_cast<char*>(output->tensor_data().data()),
                               alpha.NumElements(), stream);

      // Run PRELU with output as neg_alpha (will be overwritten with final result)
      ::musa::dnn::Binary prelu_op;
      prelu_op.SetMode(::musa::dnn::Binary::Mode::PRELU);
      auto run_status = prelu_op.Run(handle, t_output, t_input, t_output);
      OP_REQUIRES(ctx, run_status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN PRelu Run failed"));
    } else {
      // Broadcasting case: need temporary tensor for neg_alpha
      Tensor neg_alpha;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(alpha.dtype(), alpha.shape(), &neg_alpha));

      MusaNegKernelLauncher<T>(alpha.tensor_data().data(),
                               const_cast<char*>(neg_alpha.tensor_data().data()),
                               alpha.NumElements(), stream);

      mTensor t_neg_alpha = CreateMTensor(neg_alpha, format_);

      ::musa::dnn::Binary prelu_op;
      prelu_op.SetMode(::musa::dnn::Binary::Mode::PRELU);
      auto run_status = prelu_op.Run(handle, t_output, t_input, t_neg_alpha);
      OP_REQUIRES(ctx, run_status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN PRelu Run failed"));
    }

  }
};

#define REGISTER_MUSA_PRELU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("MusaPRelu").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaPReluOp<TYPE>)

REGISTER_MUSA_PRELU(float);
REGISTER_MUSA_PRELU(Eigen::half);
REGISTER_MUSA_PRELU(bfloat16);

#undef REGISTER_MUSA_PRELU

}  // namespace musa

REGISTER_OP("MusaPRelu")
    .Input("x: T")
    .Input("alpha: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Output shape matches input shape (broadcasting handled by muDNN)
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow