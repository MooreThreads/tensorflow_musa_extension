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
//   output = mask * (data_left + sliced_var_left)
//                  + (data_right + sliced_var_right)
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
    .Input(
        "data_left: T")  // data tensor (left branch, other input to left AddV2)
    .Input("sliced_var_left: T")  // StridedSlice(ReadVariableOp) output (left
                                  // bias)
    .Input("mask: T")             // Select output (gate / mask)
    .Input("data_right: T")  // data tensor (right branch, other input to right
                             // AddV2)
    .Input("sliced_var_right: T")  // StridedSlice(ReadVariableOp) output (right
                                   // bias)
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // The output shape is determined by broadcasting the data inputs
      // and mask. In practice they usually share the same shape.
      shape_inference::ShapeHandle output_shape = c->input(0);

      // Broadcast with mask (input 2)
      shape_inference::ShapeHandle mask_shape = c->input(2);
      if (c->RankKnown(output_shape) && c->RankKnown(mask_shape)) {
        TF_RETURN_IF_ERROR(c->Merge(output_shape, mask_shape, &output_shape));
      }

      c->set_output(0, output_shape);
      return Status::OK();
    });

// =============================================================================
// Kernel Implementation
// =============================================================================

template <typename T>
class MusaShiftedAffineMapOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  // Fuses 4 element-wise operations
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data_left = ctx->input(0);
    const Tensor& sliced_var_left = ctx->input(1);
    const Tensor& mask = ctx->input(2);
    const Tensor& data_right = ctx->input(3);
    const Tensor& sliced_var_right = ctx->input(4);

    VLOG(2) << "MusaShiftedAffineMap: data_left="
            << data_left.shape().DebugString()
            << ", sliced_var_left=" << sliced_var_left.shape().DebugString()
            << ", mask=" << mask.shape().DebugString()
            << ", data_right=" << data_right.shape().DebugString()
            << ", sliced_var_right=" << sliced_var_right.shape().DebugString();

    // -----------------------------------------------------------------
    // Determine output shape via BCast(data_left, mask)
    // -----------------------------------------------------------------
    BCast bcast_main(BCast::Vec(data_left.shape().dim_sizes()),
                     BCast::Vec(mask.shape().dim_sizes()));
    OP_REQUIRES(
        ctx, bcast_main.IsValid(),
        errors::InvalidArgument(
            "Incompatible shapes: data_left=", data_left.shape().DebugString(),
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
    OP_REQUIRES(
        ctx, bcast_left.IsValid(),
        errors::InvalidArgument(
            "Incompatible shapes: data_left=", data_left.shape().DebugString(),
            " vs sliced_var_left=", sliced_var_left.shape().DebugString()));

    TensorShape temp_left_shape = BCast::ToShape(bcast_left.output_shape());
    Tensor temp_left;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(data_left.dtype(), temp_left_shape,
                                           &temp_left));

    {
      mBinary add_op;
      add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mTensor mt_data = CreateMTensor(data_left, format_);
      mTensor mt_var = CreateMTensor(sliced_var_left, format_);
      mTensor mt_out = CreateMTensor(temp_left, format_);
      auto status = add_op.Run(handle, mt_out, mt_data, mt_var);
      OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                  errors::Internal("ShiftedAffineMap step1 (add left) failed: ",
                                   static_cast<int>(status)));
    }

    // -----------------------------------------------------------------
    // Step 2: temp_gated = temp_left * mask
    // -----------------------------------------------------------------
    Tensor temp_gated;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(data_left.dtype(), output_shape, &temp_gated));

    {
      mBinary mul_op;
      mul_op.SetMode(::musa::dnn::Binary::Mode::MUL);
      mTensor mt_left = CreateMTensor(temp_left, format_);
      mTensor mt_mask = CreateMTensor(mask, format_);
      mTensor mt_out = CreateMTensor(temp_gated, format_);
      auto status = mul_op.Run(handle, mt_out, mt_left, mt_mask);
      OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                  errors::Internal("ShiftedAffineMap step2 (mul mask) failed: ",
                                   static_cast<int>(status)));
    }

    // -----------------------------------------------------------------
    // Step 3: temp_right = data_right + sliced_var_right
    // -----------------------------------------------------------------
    BCast bcast_right(BCast::Vec(data_right.shape().dim_sizes()),
                      BCast::Vec(sliced_var_right.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast_right.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: data_right=",
                    data_right.shape().DebugString(), " vs sliced_var_right=",
                    sliced_var_right.shape().DebugString()));

    TensorShape temp_right_shape = BCast::ToShape(bcast_right.output_shape());
    Tensor temp_right;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(data_right.dtype(), temp_right_shape,
                                           &temp_right));

    {
      mBinary add_op;
      add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mTensor mt_data = CreateMTensor(data_right, format_);
      mTensor mt_var = CreateMTensor(sliced_var_right, format_);
      mTensor mt_out = CreateMTensor(temp_right, format_);
      auto status = add_op.Run(handle, mt_out, mt_data, mt_var);
      OP_REQUIRES(
          ctx, status == mStatus::SUCCESS,
          errors::Internal("ShiftedAffineMap step3 (add right) failed: ",
                           static_cast<int>(status)));
    }

    // -----------------------------------------------------------------
    // Step 4: output = temp_gated + temp_right
    // -----------------------------------------------------------------
    {
      mBinary add_op;
      add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
      mTensor mt_gated = CreateMTensor(temp_gated, format_);
      mTensor mt_right = CreateMTensor(temp_right, format_);
      mTensor mt_out = CreateMTensor(*output, format_);
      auto status = add_op.Run(handle, mt_out, mt_gated, mt_right);
      OP_REQUIRES(
          ctx, status == mStatus::SUCCESS,
          errors::Internal("ShiftedAffineMap step4 (final add) failed: ",
                           static_cast<int>(status)));
    }

    VLOG(2) << "MusaShiftedAffineMap: output=" << output->shape().DebugString();
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
