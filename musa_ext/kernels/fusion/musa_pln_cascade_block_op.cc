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
#include <atomic>
#include <limits>
#include <vector>

#include "../utils_op.h"
#include "kernels/math/musa_pln_cascade_block_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
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

PlnCascadeBlockShape BuildKernelShape(const TensorShape& output_shape) {
  PlnCascadeBlockShape shape{};
  shape.rank = output_shape.dims();
  for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
    shape.dims[i] = 1;
  }
  for (int i = 0; i < output_shape.dims(); ++i) {
    shape.dims[i] = static_cast<int>(output_shape.dim_size(i));
  }
  return shape;
}

PlnCascadeBlockStrides BuildBroadcastStrides(const TensorShape& input_shape,
                                             const TensorShape& output_shape) {
  PlnCascadeBlockStrides strides{};
  for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
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

PlnCascadeBlockStrides BuildLeftAligned1DStrides(
    const TensorShape& input_shape, const TensorShape& output_shape) {
  PlnCascadeBlockStrides strides{};
  for (int i = 0; i < kPlnCascadeBlockMaxDims; ++i) {
    strides.values[i] = 0;
  }

  if (input_shape.dims() == 1 && output_shape.dims() >= 1) {
    strides.values[0] = 1;
  }
  return strides;
}

bool HasIdenticalDenseLayout(const TensorShape& input_shape,
                             const TensorShape& output_shape) {
  if (input_shape.dims() != output_shape.dims()) {
    return false;
  }
  for (int i = 0; i < input_shape.dims(); ++i) {
    if (input_shape.dim_size(i) != output_shape.dim_size(i)) {
      return false;
    }
  }
  return true;
}

bool IsScalarBroadcast(const Tensor& input) { return input.NumElements() == 1; }

int ComputeInnerStride(const TensorShape& output_shape) {
  int64_t inner_stride = 1;
  for (int i = 1; i < output_shape.dims(); ++i) {
    inner_stride *= output_shape.dim_size(i);
  }
  return static_cast<int>(inner_stride);
}

}  // namespace

REGISTER_OP("MusaPlnCascadeBlock")
    .Input("norm_out: T")
    .Input("add_input: T")
    .Input("bias_input: T")
    .Input("gates: N * bool")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("N: int >= 1")
    .Attr("table_indices: list(int)")
    .Attr("select_on_true: list(bool)")
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

      ShapeHandle out = c->input(0);
      if (!c->RankKnown(out)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      for (int i = 3; i < c->num_inputs(); ++i) {
        ShapeHandle gate = c->input(i);
        if (!c->RankKnown(gate)) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }

        Status bcast_status = BroadcastTwoShapes(out, gate, &out);
        if (!bcast_status.ok()) {
          TF_RETURN_IF_ERROR(LeftAlignBatchMask(out, gate, &out));
        }
      }

      if (c->RankKnown(c->input(1)) && c->RankKnown(c->input(2))) {
        ShapeHandle merged_table;
        TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &merged_table));
      }

      c->set_output(0, out);
      return Status::OK();
    });

class MusaPlnCascadeBlockOp : public MusaOpKernel {
 public:
  explicit MusaPlnCascadeBlockOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &n_steps_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_indices", &table_indices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("select_on_true", &select_on_true_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& norm_out = ctx->input(0);
    const Tensor& add_input = ctx->input(1);
    const Tensor& bias_input = ctx->input(2);

    OpInputList gates;
    OP_REQUIRES_OK(ctx, ctx->input_list("gates", &gates));

    const int num_steps = gates.size();
    OP_REQUIRES(
        ctx, num_steps == n_steps_,
        errors::InvalidArgument("MusaPlnCascadeBlock gates size (", num_steps,
                                ") does not match attr N (", n_steps_, ")"));
    OP_REQUIRES(ctx, num_steps <= kPlnCascadeBlockMaxSteps,
                errors::InvalidArgument("MusaPlnCascadeBlock supports at most ",
                                        kPlnCascadeBlockMaxSteps,
                                        " steps, got ", num_steps));
    OP_REQUIRES(ctx, static_cast<int>(table_indices_.size()) == num_steps,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock table_indices size mismatch: ",
                    table_indices_.size(), " vs N=", num_steps));
    OP_REQUIRES(ctx, static_cast<int>(select_on_true_.size()) == num_steps,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock select_on_true size mismatch: ",
                    select_on_true_.size(), " vs N=", num_steps));

    OP_REQUIRES(ctx, add_input.dims() == 2,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock expects add_input rank 2, got ",
                    add_input.dims()));
    OP_REQUIRES(ctx, bias_input.dims() == 2,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock expects bias_input rank 2, got ",
                    bias_input.dims()));
    OP_REQUIRES(ctx, add_input.shape() == bias_input.shape(),
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock table shapes must match: add_input=",
                    add_input.shape().DebugString(),
                    ", bias_input=", bias_input.shape().DebugString()));

    const int table_rows = static_cast<int>(add_input.dim_size(0));
    const int table_width = static_cast<int>(add_input.dim_size(1));
    OP_REQUIRES(
        ctx, table_rows > 0 && table_width > 0,
        errors::InvalidArgument("MusaPlnCascadeBlock table must be non-empty"));

    for (int step = 0; step < num_steps; ++step) {
      const int table_index = table_indices_[step];
      OP_REQUIRES(ctx, table_index >= 0 && table_index < table_rows,
                  errors::InvalidArgument(
                      "MusaPlnCascadeBlock table_index out of range at step ",
                      step, ": ", table_index, ", rows=", table_rows));
    }

    TensorShape output_shape = norm_out.shape();
    std::vector<bool> gate_left_aligned(num_steps, false);
    for (int step = 0; step < num_steps; ++step) {
      const TensorShape& gate_shape = gates[step].shape();
      TensorShape bcast_shape;
      Status gate_shape_status =
          BroadcastShapes(output_shape, gate_shape, &bcast_shape);
      if (!gate_shape_status.ok()) {
        if (gate_shape.dims() == 1 && output_shape.dims() >= 2 &&
            gate_shape.dim_size(0) == output_shape.dim_size(0)) {
          gate_left_aligned[step] = true;
        } else {
          ctx->SetStatus(gate_shape_status);
          return;
        }
      } else {
        output_shape = bcast_shape;
      }
    }

    OP_REQUIRES(ctx, output_shape.dims() >= 1,
                errors::InvalidArgument(
                    "MusaPlnCascadeBlock output rank must be >= 1"));
    OP_REQUIRES(
        ctx, output_shape.dim_size(output_shape.dims() - 1) == table_width,
        errors::InvalidArgument(
            "MusaPlnCascadeBlock table width mismatch with output last dim: "
            "table_width=",
            table_width, ", output=", output_shape.DebugString()));
    OP_REQUIRES(ctx, output_shape.dims() <= kPlnCascadeBlockMaxDims,
                errors::InvalidArgument("MusaPlnCascadeBlock rank ",
                                        output_shape.dims(), " exceeds ",
                                        kPlnCascadeBlockMaxDims));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    OP_REQUIRES(
        ctx, output->NumElements() <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("MusaPlnCascadeBlock output too large: ",
                                output->NumElements()));

    const PlnCascadeBlockShape shape = BuildKernelShape(output_shape);
    const PlnCascadeBlockStrides norm_st =
        BuildBroadcastStrides(norm_out.shape(), output_shape);

    PlnCascadeBlockGatePtrs gate_ptrs{};
    PlnCascadeBlockMeta meta{};
    meta.num_steps = num_steps;
    meta.table_rows = table_rows;
    meta.table_width = table_width;
    meta.norm_is_contiguous =
        HasIdenticalDenseLayout(norm_out.shape(), output_shape) ? 1 : 0;
    meta.output_inner_stride = ComputeInnerStride(output_shape);
    meta.use_fast_path = meta.norm_is_contiguous;

    for (int step = 0; step < kPlnCascadeBlockMaxSteps; ++step) {
      gate_ptrs.values[step] = nullptr;
      meta.table_indices[step] = 0;
      meta.table_base_offsets[step] = 0;
      meta.select_on_true[step] = 1;
      meta.gate_modes[step] = kPlnCascadeBlockGateModeGeneric;
      for (int dim = 0; dim < kPlnCascadeBlockMaxDims; ++dim) {
        meta.gate_strides[step].values[dim] = 0;
      }
    }

    for (int step = 0; step < num_steps; ++step) {
      gate_ptrs.values[step] = gates[step].flat<bool>().data();
      meta.table_indices[step] = table_indices_[step];
      meta.table_base_offsets[step] = table_indices_[step] * table_width;
      meta.select_on_true[step] = select_on_true_[step] ? 1 : 0;

      const TensorShape& gate_shape = gates[step].shape();
      if (gate_left_aligned[step]) {
        meta.gate_modes[step] = kPlnCascadeBlockGateModeBatchAligned;
        meta.gate_strides[step] =
            BuildLeftAligned1DStrides(gate_shape, output_shape);
      } else if (IsScalarBroadcast(gates[step])) {
        meta.gate_modes[step] = kPlnCascadeBlockGateModeScalar;
        meta.gate_strides[step] = BuildBroadcastStrides(gate_shape, output_shape);
      } else {
        meta.use_fast_path = 0;
        meta.gate_strides[step] =
            BuildBroadcastStrides(gate_shape, output_shape);
      }
    }

    static std::atomic<int> launch_log_budget{0};
    const int launch_log_index =
        launch_log_budget.fetch_add(1, std::memory_order_relaxed);
    if (launch_log_index < 8) {
      VLOG(1) << "[PlnCascadeBlock][Kernel] launch output="
              << output_shape.DebugString() << ", num_steps=" << num_steps
              << ", fast_path=" << meta.use_fast_path
              << ", norm_contiguous=" << meta.norm_is_contiguous
              << ", table_width=" << table_width;
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchPlnCascadeBlockKernel(
        norm_out.flat<float>().data(), norm_st, gate_ptrs, meta,
        add_input.flat<float>().data(), bias_input.flat<float>().data(),
        output->flat<float>().data(), shape,
        static_cast<int>(output->NumElements()), stream);
    auto launch_status = musaGetLastError();
    OP_REQUIRES(ctx, launch_status == musaSuccess,
                errors::Internal("MUSA PlnCascadeBlock launch failed: ",
                                 musaGetErrorString(launch_status)));
  }

 private:
  int n_steps_ = 0;
  std::vector<int> table_indices_;
  std::vector<bool> select_on_true_;
};

REGISTER_KERNEL_BUILDER(
    Name("MusaPlnCascadeBlock").Device("MUSA").TypeConstraint<float>("T"),
    MusaPlnCascadeBlockOp);

}  // namespace musa
}  // namespace tensorflow
