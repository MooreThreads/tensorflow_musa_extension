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

#include <algorithm>
#include <limits>
#include <vector>

#include "../utils_op.h"
#include "kernels/math/musa_pln_cascade_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
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

PlnCascadeShape BuildKernelShape(const TensorShape& output_shape) {
  PlnCascadeShape shape{};
  shape.rank = output_shape.dims();
  for (int i = 0; i < kPlnCascadeMaxDims; ++i) {
    shape.dims[i] = 1;
  }
  for (int i = 0; i < output_shape.dims(); ++i) {
    shape.dims[i] = static_cast<int>(output_shape.dim_size(i));
  }
  return shape;
}

PlnCascadeStrides BuildBroadcastStrides(const TensorShape& input_shape,
                                        const TensorShape& output_shape) {
  PlnCascadeStrides strides{};
  for (int i = 0; i < kPlnCascadeMaxDims; ++i) {
    strides.values[i] = 0;
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
      strides.values[out_axis] = 0;
      continue;
    }

    if (input_shape.dim_size(in_axis) == 1 &&
        output_shape.dim_size(out_axis) > 1) {
      strides.values[out_axis] = 0;
    } else {
      strides.values[out_axis] = static_cast<int>(dense_strides[in_axis]);
    }
  }

  return strides;
}

PlnCascadeStrides BuildLeftAligned1DStrides(const TensorShape& input_shape,
                                            const TensorShape& output_shape) {
  PlnCascadeStrides strides{};
  for (int i = 0; i < kPlnCascadeMaxDims; ++i) {
    strides.values[i] = 0;
  }

  if (input_shape.dims() == 1 && output_shape.dims() >= 1) {
    strides.values[0] = 1;
  }
  return strides;
}

}  // namespace

REGISTER_OP("MusaPlnCascade")
    .Input("norm_out: T")
    .Input("adpos: bool")
    .Input("add_input: T")
    .Input("bias_input: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_table: bool = false")
    .Attr("table_index: int = 0")
    .Attr("select_on_true: bool = true")
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

      auto LeftAlignBatchMask = [&](ShapeHandle base, ShapeHandle mask,
                                    ShapeHandle* out) -> Status {
        const int rank_base = c->Rank(base);
        const int rank_mask = c->Rank(mask);
        if (rank_base < 2 || rank_mask != 1) {
          return errors::InvalidArgument(
              "Incompatible shapes for left-aligned mask broadcast");
        }

        DimensionHandle merged_dim0;
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(base, 0), c->Dim(mask, 0), &merged_dim0));

        std::vector<DimensionHandle> dims;
        dims.reserve(rank_base);
        dims.push_back(merged_dim0);
        for (int i = 1; i < rank_base; ++i) {
          dims.push_back(c->Dim(base, i));
        }
        *out = c->MakeShape(dims);
        return Status::OK();
      };

      bool use_table = false;
      TF_RETURN_IF_ERROR(c->GetAttr("use_table", &use_table));

      ShapeHandle out = c->input(0);
      if (!c->RankKnown(out) || !c->RankKnown(c->input(1))) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      Status mask_bcast_status = BroadcastTwoShapes(out, c->input(1), &out);
      if (!mask_bcast_status.ok()) {
        TF_RETURN_IF_ERROR(LeftAlignBatchMask(out, c->input(1), &out));
      }
      if (!use_table) {
        if (!c->RankKnown(c->input(2)) || !c->RankKnown(c->input(3))) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(BroadcastTwoShapes(out, c->input(2), &out));
        TF_RETURN_IF_ERROR(BroadcastTwoShapes(out, c->input(3), &out));
      }

      c->set_output(0, out);
      return Status::OK();
    });

class MusaPlnCascadeOp : public MusaOpKernel {
 public:
  explicit MusaPlnCascadeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_table", &use_table_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_index", &table_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_on_true", &select_on_true_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& norm_out = ctx->input(0);
    const Tensor& adpos = ctx->input(1);
    const Tensor& add_input = ctx->input(2);
    const Tensor& bias_input = ctx->input(3);

    TensorShape output_shape;
    bool adpos_left_aligned = false;
    Status adpos_shape_status =
        BroadcastShapes(norm_out.shape(), adpos.shape(), &output_shape);
    if (!adpos_shape_status.ok()) {
      if (adpos.dims() == 1 && norm_out.dims() >= 2 &&
          adpos.dim_size(0) == norm_out.dim_size(0)) {
        output_shape = norm_out.shape();
        adpos_left_aligned = true;
      } else {
        ctx->SetStatus(adpos_shape_status);
        return;
      }
    }

    int table_rows = 0;
    int table_width = 0;
    if (use_table_) {
      OP_REQUIRES(
          ctx, add_input.dims() == 2,
          errors::InvalidArgument(
              "MusaPlnCascade(use_table=true) expects add_input rank 2, "
              "got ",
              add_input.dims()));
      OP_REQUIRES(
          ctx, bias_input.dims() == 2,
          errors::InvalidArgument(
              "MusaPlnCascade(use_table=true) expects bias_input rank 2, "
              "got ",
              bias_input.dims()));
      OP_REQUIRES(ctx, add_input.shape() == bias_input.shape(),
                  errors::InvalidArgument(
                      "MusaPlnCascade table shapes must match: add_input=",
                      add_input.shape().DebugString(),
                      ", bias_input=", bias_input.shape().DebugString()));

      table_rows = static_cast<int>(add_input.dim_size(0));
      table_width = static_cast<int>(add_input.dim_size(1));

      OP_REQUIRES(
          ctx, table_rows > 0 && table_width > 0,
          errors::InvalidArgument("MusaPlnCascade table must be non-empty"));
      OP_REQUIRES(ctx, table_index_ >= 0 && table_index_ < table_rows,
                  errors::InvalidArgument(
                      "MusaPlnCascade table_index out of range: ", table_index_,
                      ", rows=", table_rows));

      OP_REQUIRES(
          ctx, output_shape.dims() >= 1,
          errors::InvalidArgument(
              "MusaPlnCascade(use_table=true) output rank must be >= 1"));
      OP_REQUIRES(
          ctx, output_shape.dim_size(output_shape.dims() - 1) == table_width,
          errors::InvalidArgument(
              "MusaPlnCascade table width mismatch with output last dim: "
              "table_width=",
              table_width, ", output=", output_shape.DebugString()));
    } else {
      TensorShape tmp_shape;
      OP_REQUIRES_OK(
          ctx, BroadcastShapes(output_shape, add_input.shape(), &tmp_shape));
      OP_REQUIRES_OK(
          ctx, BroadcastShapes(tmp_shape, bias_input.shape(), &output_shape));
    }

    OP_REQUIRES(
        ctx, output_shape.dims() <= kPlnCascadeMaxDims,
        errors::InvalidArgument("MusaPlnCascade rank ", output_shape.dims(),
                                " exceeds ", kPlnCascadeMaxDims));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    OP_REQUIRES(ctx, output->NumElements() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("MusaPlnCascade output too large: ",
                                        output->NumElements()));

    const PlnCascadeShape shape = BuildKernelShape(output_shape);
    const PlnCascadeStrides norm_st =
        BuildBroadcastStrides(norm_out.shape(), output_shape);
    const PlnCascadeStrides adpos_st =
        adpos_left_aligned
            ? BuildLeftAligned1DStrides(adpos.shape(), output_shape)
            : BuildBroadcastStrides(adpos.shape(), output_shape);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    if (use_table_) {
      LaunchPlnCascadeTableKernel(
          norm_out.flat<float>().data(), norm_st, adpos.flat<bool>().data(),
          adpos_st, add_input.flat<float>().data(),
          bias_input.flat<float>().data(), table_rows, table_width,
          table_index_, output->flat<float>().data(), shape,
          static_cast<int>(output->NumElements()), select_on_true_, stream);
    } else {
      const PlnCascadeStrides add_st =
          BuildBroadcastStrides(add_input.shape(), output_shape);
      const PlnCascadeStrides bias_st =
          BuildBroadcastStrides(bias_input.shape(), output_shape);

      LaunchPlnCascadeDirectKernel(
          norm_out.flat<float>().data(), norm_st, adpos.flat<bool>().data(),
          adpos_st, add_input.flat<float>().data(), add_st,
          bias_input.flat<float>().data(), bias_st,
          output->flat<float>().data(), shape,
          static_cast<int>(output->NumElements()), select_on_true_, stream);
    }
    auto launch_status = musaGetLastError();
    OP_REQUIRES(ctx, launch_status == musaSuccess,
                errors::Internal("MUSA PlnCascade launch failed: ",
                                 musaGetErrorString(launch_status)));
  }

 private:
  bool use_table_ = false;
  int table_index_ = 0;
  bool select_on_true_ = true;
};

REGISTER_KERNEL_BUILDER(
    Name("MusaPlnCascade").Device("MUSA").TypeConstraint<float>("T"),
    MusaPlnCascadeOp);

}  // namespace musa
}  // namespace tensorflow
