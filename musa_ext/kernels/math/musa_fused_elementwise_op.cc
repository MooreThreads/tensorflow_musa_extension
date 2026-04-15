#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../utils_op.h"
#include "kernels/math/musa_fused_elementwise_kernel.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/bcast.h"

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

Status ComputeOutputShape(OpKernelContext* ctx, TensorShape* output_shape) {
  if (ctx->num_inputs() <= 0) {
    return errors::InvalidArgument(
        "MusaFusedElementwise requires at least one input");
  }
  *output_shape = ctx->input(0).shape();
  for (int i = 1; i < ctx->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(
        BroadcastShapes(*output_shape, ctx->input(i).shape(), output_shape));
  }
  return Status::OK();
}

MusaFusedElementwiseConfig BuildKernelConfig(
    const std::vector<TensorShape>& input_shapes,
    const TensorShape& output_shape, const std::vector<int>& opcodes,
    const std::vector<int>& step_arities, const std::vector<int>& arg0_kinds,
    const std::vector<int>& arg0_inputs, const std::vector<int>& arg1_kinds,
    const std::vector<int>& arg1_inputs) {
  MusaFusedElementwiseConfig config{};
  config.rank = output_shape.dims();
  config.num_inputs = static_cast<int>(input_shapes.size());
  config.num_steps = static_cast<int>(opcodes.size());

  for (int dim = 0; dim < kMusaFusedElementwiseMaxDims; ++dim) {
    config.dims[dim] = 1;
  }
  for (int dim = 0; dim < output_shape.dims(); ++dim) {
    config.dims[dim] = static_cast<int>(output_shape.dim_size(dim));
  }

  for (int input_idx = 0; input_idx < config.num_inputs; ++input_idx) {
    for (int dim = 0; dim < kMusaFusedElementwiseMaxDims; ++dim) {
      config.input_strides[input_idx][dim] = 0;
    }

    const TensorShape& input_shape = input_shapes[input_idx];
    std::vector<int64_t> dense_strides(input_shape.dims(), 1);
    int64_t acc = 1;
    for (int dim = input_shape.dims() - 1; dim >= 0; --dim) {
      dense_strides[dim] = acc;
      acc *= input_shape.dim_size(dim);
    }

    const int rank_delta = output_shape.dims() - input_shape.dims();
    for (int out_axis = 0; out_axis < output_shape.dims(); ++out_axis) {
      const int in_axis = out_axis - rank_delta;
      if (in_axis < 0) {
        config.input_strides[input_idx][out_axis] = 0;
        continue;
      }

      if (input_shape.dim_size(in_axis) == 1 &&
          output_shape.dim_size(out_axis) > 1) {
        config.input_strides[input_idx][out_axis] = 0;
      } else {
        config.input_strides[input_idx][out_axis] =
            static_cast<int>(dense_strides[in_axis]);
      }
    }
  }

  for (int step = 0; step < config.num_steps; ++step) {
    config.step_opcode[step] = opcodes[step];
    config.step_arity[step] = step_arities[step];
    config.step_arg_kind[step][0] = arg0_kinds[step];
    config.step_arg_input[step][0] = arg0_inputs[step];
    config.step_arg_kind[step][1] = arg1_kinds[step];
    config.step_arg_input[step][1] = arg1_inputs[step];
  }

  return config;
}

bool IsValidOperand(int operand_kind, int operand_input, int num_inputs) {
  if (operand_kind == kOperandPrev) {
    return operand_input == -1;
  }
  if (operand_kind == kOperandInput) {
    return operand_input >= 0 && operand_input < num_inputs;
  }
  return operand_kind == kOperandNone && operand_input == -1;
}

}  // namespace

REGISTER_OP("MusaFusedElementwise")
    .Input("inputs: num_inputs * T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("num_inputs: int >= 1")
    .Attr("opcodes: list(int)")
    .Attr("step_arities: list(int)")
    .Attr("arg0_kinds: list(int)")
    .Attr("arg0_inputs: list(int)")
    .Attr("arg1_kinds: list(int)")
    .Attr("arg1_inputs: list(int)")
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

      if (c->num_inputs() <= 0) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      ShapeHandle out = c->input(0);
      if (!c->RankKnown(out)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      for (int i = 1; i < c->num_inputs(); ++i) {
        if (!c->RankKnown(c->input(i))) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(BroadcastTwoShapes(out, c->input(i), &out));
      }

      c->set_output(0, out);
      return Status::OK();
    });

template <typename T>
class MusaFusedElementwiseOp : public MusaOpKernel {
 public:
  explicit MusaFusedElementwiseOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_inputs", &num_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("opcodes", &opcodes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("step_arities", &step_arities_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg0_kinds", &arg0_kinds_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg0_inputs", &arg0_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg1_kinds", &arg1_kinds_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg1_inputs", &arg1_inputs_));

    const size_t num_steps = opcodes_.size();
    OP_REQUIRES(ctx, num_inputs_ >= 1,
                errors::InvalidArgument("num_inputs must be >= 1"));
    OP_REQUIRES(ctx, num_steps > 0,
                errors::InvalidArgument("MusaFusedElementwise requires at "
                                        "least one step"));
    OP_REQUIRES(
        ctx,
        num_steps == step_arities_.size() && num_steps == arg0_kinds_.size() &&
            num_steps == arg0_inputs_.size() &&
            num_steps == arg1_kinds_.size() &&
            num_steps == arg1_inputs_.size(),
        errors::InvalidArgument(
            "MusaFusedElementwise attr list sizes must match"));
    OP_REQUIRES(ctx, num_inputs_ <= kMusaFusedElementwiseMaxInputs,
                errors::InvalidArgument("num_inputs exceeds kernel limit"));
    OP_REQUIRES(ctx, num_steps <= kMusaFusedElementwiseMaxSteps,
                errors::InvalidArgument("num_steps exceeds kernel limit"));

    for (size_t i = 0; i < num_steps; ++i) {
      OP_REQUIRES(ctx, step_arities_[i] == 1 || step_arities_[i] == 2,
                  errors::InvalidArgument("Unsupported step arity at step ",
                                          i, ": ", step_arities_[i]));
      OP_REQUIRES(ctx,
                  IsValidOperand(arg0_kinds_[i], arg0_inputs_[i], num_inputs_),
                  errors::InvalidArgument("Invalid arg0 operand encoding at "
                                          "step ",
                                          i));
      if (step_arities_[i] == 1) {
        OP_REQUIRES(
            ctx, IsValidOperand(arg1_kinds_[i], arg1_inputs_[i], num_inputs_) &&
                     arg1_kinds_[i] == kOperandNone,
            errors::InvalidArgument("Unary step ", i,
                                    " must leave arg1 unused"));
      } else {
        OP_REQUIRES(
            ctx, IsValidOperand(arg1_kinds_[i], arg1_inputs_[i], num_inputs_),
            errors::InvalidArgument("Invalid arg1 operand encoding at step ",
                                    i));
      }
    }
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == num_inputs_,
                errors::InvalidArgument("Expected ", num_inputs_,
                                        " inputs, got ", ctx->num_inputs()));

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, ComputeOutputShape(ctx, &output_shape));
    OP_REQUIRES(ctx, output_shape.dims() <= kMusaFusedElementwiseMaxDims,
                errors::InvalidArgument(
                    "MusaFusedElementwise rank exceeds kernel limit: ",
                    output_shape.dims()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    OP_REQUIRES(
        ctx, output->NumElements() <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("MusaFusedElementwise output is too large for "
                                "single-kernel indexing: ",
                                output->NumElements()));

    std::vector<TensorShape> input_shapes;
    input_shapes.reserve(ctx->num_inputs());

    MusaFusedElementwiseInlinePointers input_ptrs{};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& input = ctx->input(i);
      input_shapes.push_back(input.shape());
      input_ptrs.ptrs[i] = input.tensor_data().data();
    }

    const MusaFusedElementwiseConfig config = BuildKernelConfig(
        input_shapes, output_shape, opcodes_, step_arities_, arg0_kinds_,
        arg0_inputs_, arg1_kinds_, arg1_inputs_);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("MUSA stream is null"));

    LaunchMusaFusedElementwiseKernel<T>(
        input_ptrs, output->flat<T>().data(), config,
        static_cast<int>(output->NumElements()), stream);

    const musaError_t launch_status = musaGetLastError();
    OP_REQUIRES(ctx, launch_status == musaSuccess,
                errors::Internal(
                    "MusaFusedElementwise kernel launch failed: ",
                    musaGetErrorString(launch_status)));
  }

 private:
  int num_inputs_ = 0;
  std::vector<int> opcodes_;
  std::vector<int> step_arities_;
  std::vector<int> arg0_kinds_;
  std::vector<int> arg0_inputs_;
  std::vector<int> arg1_kinds_;
  std::vector<int> arg1_inputs_;
};

#define REGISTER_MUSA_FUSED_ELEMENTWISE(TYPE)                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("MusaFusedElementwise").Device("MUSA").TypeConstraint<TYPE>( \
          "T"),                                                        \
      MusaFusedElementwiseOp<TYPE>);

REGISTER_MUSA_FUSED_ELEMENTWISE(float);
REGISTER_MUSA_FUSED_ELEMENTWISE(double);
REGISTER_MUSA_FUSED_ELEMENTWISE(Eigen::half);
REGISTER_MUSA_FUSED_ELEMENTWISE(bfloat16);

#undef REGISTER_MUSA_FUSED_ELEMENTWISE

}  // namespace musa
}  // namespace tensorflow
