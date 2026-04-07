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

#include <algorithm>
#include <limits>
#include <vector>

#include "../utils_op.h"
#include "kernels/math/musa_shifted_affine_map_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

Status BroadcastShapes(const TensorShape& lhs, const TensorShape& rhs,
                       TensorShape* output) {
  BCast bcast(BCast::Vec(lhs.dim_sizes()), BCast::Vec(rhs.dim_sizes()));
  if (!bcast.IsValid()) {
    return errors::InvalidArgument("Incompatible shapes: ", lhs.DebugString(),
                                   " vs ", rhs.DebugString());
  }
  *output = BCast::ToShape(bcast.output_shape());
  return Status::OK();
}

ShiftedAffineMapShape BuildKernelShape(const TensorShape& output_shape) {
  ShiftedAffineMapShape kernel_shape{};
  kernel_shape.rank = output_shape.dims();
  for (int i = 0; i < kShiftedAffineMapMaxDims; ++i) {
    kernel_shape.dims[i] = 1;
  }
  for (int i = 0; i < output_shape.dims(); ++i) {
    kernel_shape.dims[i] = static_cast<int>(output_shape.dim_size(i));
  }
  return kernel_shape;
}

ShiftedAffineMapStrides BuildBroadcastStrides(const TensorShape& input_shape,
                                              const TensorShape& output_shape) {
  ShiftedAffineMapStrides kernel_strides{};
  for (int i = 0; i < kShiftedAffineMapMaxDims; ++i) {
    kernel_strides.values[i] = 0;
  }

  std::vector<int64_t> dense_strides(input_shape.dims(), 1);
  int64_t acc = 1;
  for (int i = input_shape.dims() - 1; i >= 0; --i) {
    dense_strides[i] = acc;
    acc *= input_shape.dim_size(i);
  }

  const int rank_delta = output_shape.dims() - input_shape.dims();
  for (int out_axis = 0; out_axis < output_shape.dims(); ++out_axis) {
    const int in_axis = out_axis - rank_delta;
    if (in_axis < 0) {
      kernel_strides.values[out_axis] = 0;
      continue;
    }

    if (input_shape.dim_size(in_axis) == 1 &&
        output_shape.dim_size(out_axis) > 1) {
      kernel_strides.values[out_axis] = 0;
    } else {
      kernel_strides.values[out_axis] =
          static_cast<int>(dense_strides[in_axis]);
    }
  }

  return kernel_strides;
}

}  // namespace

// =============================================================================
// Op Registration
// =============================================================================

REGISTER_OP("MusaShiftedAffineMap")
    .Input("data_left: T")  // other input of inner AddV2/Mul
    .Input("mask: T")       // Select output (gate)
    .Input(
        "sliced_var_right: T")  // StridedSlice(ReadVariableOp) — right addend
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using ::tensorflow::shape_inference::DimensionHandle;
      using ::tensorflow::shape_inference::ShapeHandle;

      auto BroadcastTwoShapes = [&](ShapeHandle a, ShapeHandle b,
                                    ShapeHandle* out) -> Status {
        const int rank_a = c->Rank(a);
        const int rank_b = c->Rank(b);
        const int out_rank = std::max(rank_a, rank_b);

        std::vector<DimensionHandle> dims;
        dims.reserve(out_rank);

        for (int i = 0; i < out_rank; ++i) {
          const int ia = rank_a - 1 - i;
          const int ib = rank_b - 1 - i;

          auto dim_a = (ia >= 0) ? c->Dim(a, ia) : c->MakeDim(1);
          auto dim_b = (ib >= 0) ? c->Dim(b, ib) : c->MakeDim(1);

          if (c->ValueKnown(dim_a) && c->Value(dim_a) == 1) {
            dims.push_back(dim_b);
            continue;
          }
          if (c->ValueKnown(dim_b) && c->Value(dim_b) == 1) {
            dims.push_back(dim_a);
            continue;
          }

          DimensionHandle merged;
          TF_RETURN_IF_ERROR(c->Merge(dim_a, dim_b, &merged));
          dims.push_back(merged);
        }

        std::reverse(dims.begin(), dims.end());
        *out = c->MakeShape(dims);
        return Status::OK();
      };

      ShapeHandle out = c->input(0);
      if (!c->RankKnown(out) || !c->RankKnown(c->input(1)) ||
          !c->RankKnown(c->input(2))) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      TF_RETURN_IF_ERROR(BroadcastTwoShapes(out, c->input(1), &out));
      TF_RETURN_IF_ERROR(BroadcastTwoShapes(out, c->input(2), &out));
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
    MUSA_KERNEL_TIMING_GUARD(ctx);

    const Tensor& data_left = ctx->input(0);
    const Tensor& mask = ctx->input(1);
    const Tensor& sliced_var_right = ctx->input(2);

    // =========================================================================
    // FAST PATH: Check if all inputs have exactly the same shape.
    // If so, we can bypass complex stride calculations and use a fast 1D
    // kernel.
    // =========================================================================
    bool is_same_shape = (data_left.shape() == mask.shape()) &&
                         (data_left.shape() == sliced_var_right.shape());

    if (is_same_shape) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, data_left.shape(), &output));
      if (output->NumElements() == 0) return;

      musaStream_t stream = GetMusaStreamByCtx(ctx);
      MUSA_KERNEL_TRACE_START("Kernel_FastPath");
      LaunchShiftedAffineMapContiguous<T>(
          data_left.flat<T>().data(), mask.flat<T>().data(),
          sliced_var_right.flat<T>().data(), output->flat<T>().data(),
          static_cast<int64_t>(output->NumElements()), stream);
      MUSA_KERNEL_TRACE_END("Kernel_FastPath");

      auto launch_status = musaGetLastError();
      OP_REQUIRES(
          ctx, launch_status == musaSuccess,
          errors::Internal("MUSA ShiftedAffineMap fast path launch failed: ",
                           musaGetErrorString(launch_status)));

      VLOG(2) << "MusaShiftedAffineMap (FastPath) output="
              << output->shape().DebugString();
      return;
    }

    VLOG(2) << "MusaShiftedAffineMap:"
            << " data_left=" << data_left.shape().DebugString()
            << " mask=" << mask.shape().DebugString()
            << " sliced_var_right=" << sliced_var_right.shape().DebugString();

    TensorShape temp_left_shape;
    OP_REQUIRES_OK(ctx, BroadcastShapes(data_left.shape(), mask.shape(),
                                        &temp_left_shape));

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   BroadcastShapes(temp_left_shape, sliced_var_right.shape(),
                                   &output_shape));

    OP_REQUIRES(ctx, output_shape.dims() <= kShiftedAffineMapMaxDims,
                errors::InvalidArgument(
                    "ShiftedAffineMap rank ", output_shape.dims(),
                    " exceeds kernel limit ", kShiftedAffineMapMaxDims));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    OP_REQUIRES(
        ctx, output->NumElements() <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("ShiftedAffineMap output is too large for "
                                "single-kernel indexing: ",
                                output->NumElements()));

    ShiftedAffineMapShape kernel_shape = BuildKernelShape(output_shape);
    ShiftedAffineMapStrides data_left_st =
        BuildBroadcastStrides(data_left.shape(), output_shape);
    ShiftedAffineMapStrides mask_st =
        BuildBroadcastStrides(mask.shape(), output_shape);
    ShiftedAffineMapStrides sliced_var_right_st =
        BuildBroadcastStrides(sliced_var_right.shape(), output_shape);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    MUSA_KERNEL_TRACE_START("Kernel");
    LaunchShiftedAffineMapKernel<T>(
        data_left.flat<T>().data(), data_left_st, mask.flat<T>().data(),
        mask_st, sliced_var_right.flat<T>().data(), sliced_var_right_st,
        output->flat<T>().data(), kernel_shape,
        static_cast<int>(output->NumElements()), stream);
    MUSA_KERNEL_TRACE_END("Kernel");

    auto launch_status = musaGetLastError();
    OP_REQUIRES(
        ctx, launch_status == musaSuccess,
        errors::Internal("MUSA ShiftedAffineMap broadcast path launch failed: ",
                         musaGetErrorString(launch_status)));

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
