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

//
// MusaShiftedAffineMap custom Op / Kernel
//
// Computes:
//   output = mask * (data_left + sliced_var_left) + sliced_var_right
//
// All operations are element-wise with broadcasting support.
//

#include <mudnn.h>

#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace musa {

// =============================================================================
// Op Registration
// =============================================================================

REGISTER_OP("MusaShiftedAffineMap")
    .Input("data_left: T")        // other input of inner AddV2 (left branch)
    .Input("sliced_var_left: T")  // StridedSlice(ReadVariableOp) — left bias
    .Input("mask: T")             // Select output (gate)
    .Input("sliced_var_right: T") // StridedSlice(ReadVariableOp) — right addend
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Output shape = broadcast(data_left, mask)
      shape_inference::ShapeHandle out = c->input(0);
      shape_inference::ShapeHandle mask_sh = c->input(2);
      if (c->RankKnown(out) && c->RankKnown(mask_sh))
        TF_RETURN_IF_ERROR(c->Merge(out, mask_sh, &out));
      c->set_output(0, out);
      return Status::OK();
    });

// =============================================================================
// Kernel Implementation
// =============================================================================

template <typename T>
class MusaShiftedAffineMapOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data_left      = ctx->input(0);
    const Tensor& sliced_var_left = ctx->input(1);
    const Tensor& mask            = ctx->input(2);
    const Tensor& sliced_var_right = ctx->input(3);

    VLOG(2) << "MusaShiftedAffineMap:"
            << " data_left=" << data_left.shape().DebugString()
            << " sliced_var_left=" << sliced_var_left.shape().DebugString()
            << " mask=" << mask.shape().DebugString()
            << " sliced_var_right=" << sliced_var_right.shape().DebugString();

    // -----------------------------------------------------------------
    // Output shape = BCast(data_left, mask)
    // -----------------------------------------------------------------
    BCast bcast_main(BCast::Vec(data_left.shape().dim_sizes()),
                     BCast::Vec(mask.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast_main.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: data_left=",
                    data_left.shape().DebugString(),
                    " vs mask=", mask.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast_main.output_shape());
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    // -----------------------------------------------------------------
    // Step 1: temp_left = data_left + sliced_var_left
    // -----------------------------------------------------------------
    BCast bcast_left(BCast::Vec(data_left.shape().dim_sizes()),
                     BCast::Vec(sliced_var_left.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast_left.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: data_left=",
                    data_left.shape().DebugString(),
                    " vs sliced_var_left=",
                    sliced_var_left.shape().DebugString()));

    Tensor temp_left;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        data_left.dtype(), BCast::ToShape(bcast_left.output_shape()),
        &temp_left));
    {
      mBinary op; op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mTensor temp_left_mt = CreateMTensor(temp_left, format_);
      mTensor data_left_mt = CreateMTensor(data_left, format_);
      mTensor sliced_var_left_mt = CreateMTensor(sliced_var_left, format_);
      auto s = op.Run(handle,
                      temp_left_mt,
                      data_left_mt,
                      sliced_var_left_mt);
      OP_REQUIRES(ctx, s == mStatus::SUCCESS,
                  errors::Internal("ShiftedAffineMap ADD left failed: ",
                                   static_cast<int>(s)));
    }

    // -----------------------------------------------------------------
    // Step 2: temp_gated = temp_left * mask
    // -----------------------------------------------------------------
    Tensor temp_gated;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        data_left.dtype(), output_shape, &temp_gated));
    {
      mBinary op; op.SetMode(::musa::dnn::Binary::Mode::MUL);
      mTensor temp_gated_mt = CreateMTensor(temp_gated, format_);
      mTensor temp_left_mt = CreateMTensor(temp_left, format_);
      mTensor mask_mt = CreateMTensor(mask, format_);
      auto s = op.Run(handle,
                      temp_gated_mt,
                      temp_left_mt,
                      mask_mt);
      OP_REQUIRES(ctx, s == mStatus::SUCCESS,
                  errors::Internal("ShiftedAffineMap MUL mask failed: ",
                                   static_cast<int>(s)));
    }

    // -----------------------------------------------------------------
    // Step 3: output = temp_gated + sliced_var_right
    // -----------------------------------------------------------------
    {
      mBinary op; op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mTensor output_mt = CreateMTensor(*output, format_);
      mTensor temp_gated_mt = CreateMTensor(temp_gated, format_);
      mTensor sliced_var_right_mt = CreateMTensor(sliced_var_right, format_);
      auto s = op.Run(handle,
                      output_mt,
                      temp_gated_mt,
                      sliced_var_right_mt);
      OP_REQUIRES(ctx, s == mStatus::SUCCESS,
                  errors::Internal("ShiftedAffineMap ADD right failed: ",
                                   static_cast<int>(s)));
    }

    VLOG(2) << "MusaShiftedAffineMap output=" << output->shape().DebugString();
  }
};

// =============================================================================
// Kernel Registration
// =============================================================================

#define REGISTER_MUSA_SHIFTED_AFFINE_MAP(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MusaShiftedAffineMap").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaShiftedAffineMapOp<TYPE>);

REGISTER_MUSA_SHIFTED_AFFINE_MAP(float);
REGISTER_MUSA_SHIFTED_AFFINE_MAP(double);
REGISTER_MUSA_SHIFTED_AFFINE_MAP(Eigen::half);
REGISTER_MUSA_SHIFTED_AFFINE_MAP(bfloat16);

#undef REGISTER_MUSA_SHIFTED_AFFINE_MAP

}  // namespace musa
}  // namespace tensorflow
