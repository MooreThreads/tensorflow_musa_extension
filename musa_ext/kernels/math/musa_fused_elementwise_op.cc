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

void FillBroadcastStrides(const TensorShape& input_shape,
                          const TensorShape& output_shape,
                          int* target_strides) {
  for (int dim = 0; dim < kMusaFusedElementwiseMaxDims; ++dim) {
    target_strides[dim] = 0;
  }

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
      target_strides[out_axis] = 0;
      continue;
    }

    if (input_shape.dim_size(in_axis) == 1 &&
        output_shape.dim_size(out_axis) > 1) {
      target_strides[out_axis] = 0;
    } else {
      target_strides[out_axis] = static_cast<int>(dense_strides[in_axis]);
    }
  }
}

MusaFusedElementwiseConfig BuildKernelConfig(
    const std::vector<TensorShape>& data_input_shapes,
    const std::vector<TensorShape>& bool_input_shapes,
    const TensorShape& output_shape, const std::vector<int>& opcodes,
    const std::vector<int>& step_arities, const std::vector<int>& arg0_kinds,
    const std::vector<int>& arg0_inputs, const std::vector<int>& arg1_kinds,
    const std::vector<int>& arg1_inputs, const std::vector<int>& arg2_kinds,
    const std::vector<int>& arg2_inputs) {
  MusaFusedElementwiseConfig config{};
  config.rank = output_shape.dims();
  config.num_data_inputs = static_cast<int>(data_input_shapes.size());
  config.num_bool_inputs = static_cast<int>(bool_input_shapes.size());
  config.num_steps = static_cast<int>(opcodes.size());

  for (int dim = 0; dim < kMusaFusedElementwiseMaxDims; ++dim) {
    config.dims[dim] = 1;
  }
  for (int dim = 0; dim < output_shape.dims(); ++dim) {
    config.dims[dim] = static_cast<int>(output_shape.dim_size(dim));
  }

  for (int input_idx = 0; input_idx < config.num_data_inputs; ++input_idx) {
    FillBroadcastStrides(data_input_shapes[input_idx], output_shape,
                         config.data_input_strides[input_idx]);
  }
  for (int input_idx = 0; input_idx < config.num_bool_inputs; ++input_idx) {
    FillBroadcastStrides(bool_input_shapes[input_idx], output_shape,
                         config.bool_input_strides[input_idx]);
  }

  for (int step = 0; step < config.num_steps; ++step) {
    config.step_opcode[step] = opcodes[step];
    config.step_arity[step] = step_arities[step];
    config.step_arg_kind[step][0] = arg0_kinds[step];
    config.step_arg_input[step][0] = arg0_inputs[step];
    config.step_arg_kind[step][1] = arg1_kinds[step];
    config.step_arg_input[step][1] = arg1_inputs[step];
    config.step_arg_kind[step][2] = arg2_kinds[step];
    config.step_arg_input[step][2] = arg2_inputs[step];
  }

  return config;
}

bool IsValidOperand(int operand_kind, int operand_input, int num_data_inputs,
                    int num_bool_inputs, int num_steps, int step_index) {
  if (operand_kind == kOperandDataInput) {
    return operand_input >= 0 && operand_input < num_data_inputs;
  }
  if (operand_kind == kOperandBoolInput) {
    return operand_input >= 0 && operand_input < num_bool_inputs;
  }
  if (operand_kind == kOperandStep) {
    return operand_input >= 0 && operand_input < num_steps &&
           operand_input < step_index;
  }
  return operand_kind == kOperandNone && operand_input == -1;
}

}  // namespace

REGISTER_OP("MusaFusedElementwise")
    .Input("data_inputs: num_data_inputs * T")
    .Input("bool_inputs: num_bool_inputs * bool")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("num_data_inputs: int >= 1")
    .Attr("num_bool_inputs: int >= 0 = 0")
    .Attr("opcodes: list(int)")
    .Attr("step_arities: list(int)")
    .Attr("arg0_kinds: list(int)")
    .Attr("arg0_inputs: list(int)")
    .Attr("arg1_kinds: list(int)")
    .Attr("arg1_inputs: list(int)")
    .Attr("arg2_kinds: list(int)")
    .Attr("arg2_inputs: list(int)")
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_data_inputs", &num_data_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bool_inputs", &num_bool_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("opcodes", &opcodes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("step_arities", &step_arities_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg0_kinds", &arg0_kinds_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg0_inputs", &arg0_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg1_kinds", &arg1_kinds_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg1_inputs", &arg1_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg2_kinds", &arg2_kinds_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("arg2_inputs", &arg2_inputs_));

    const size_t num_steps = opcodes_.size();
    OP_REQUIRES(ctx, num_data_inputs_ >= 1,
                errors::InvalidArgument("num_data_inputs must be >= 1"));
    OP_REQUIRES(ctx, num_bool_inputs_ >= 0,
                errors::InvalidArgument("num_bool_inputs must be >= 0"));
    OP_REQUIRES(ctx, num_steps > 0,
                errors::InvalidArgument("MusaFusedElementwise requires at "
                                        "least one step"));
    OP_REQUIRES(
        ctx,
        num_steps == step_arities_.size() && num_steps == arg0_kinds_.size() &&
            num_steps == arg0_inputs_.size() &&
            num_steps == arg1_kinds_.size() &&
            num_steps == arg1_inputs_.size() &&
            num_steps == arg2_kinds_.size() &&
            num_steps == arg2_inputs_.size(),
        errors::InvalidArgument(
            "MusaFusedElementwise attr list sizes must match"));
    OP_REQUIRES(
        ctx, num_data_inputs_ <= kMusaFusedElementwiseMaxDataInputs,
        errors::InvalidArgument("num_data_inputs exceeds kernel limit"));
    OP_REQUIRES(
        ctx, num_bool_inputs_ <= kMusaFusedElementwiseMaxBoolInputs,
        errors::InvalidArgument("num_bool_inputs exceeds kernel limit"));
    OP_REQUIRES(ctx, num_steps <= kMusaFusedElementwiseMaxSteps,
                errors::InvalidArgument("num_steps exceeds kernel limit"));

    for (size_t i = 0; i < num_steps; ++i) {
      OP_REQUIRES(ctx,
                  step_arities_[i] >= 1 &&
                      step_arities_[i] <= kMusaFusedElementwiseMaxArity,
                  errors::InvalidArgument("Unsupported step arity at step ",
                                          i, ": ", step_arities_[i]));
      OP_REQUIRES(
          ctx,
          IsValidOperand(arg0_kinds_[i], arg0_inputs_[i], num_data_inputs_,
                         num_bool_inputs_, static_cast<int>(num_steps),
                         static_cast<int>(i)),
          errors::InvalidArgument("Invalid arg0 operand encoding at step ",
                                  i));

      if (step_arities_[i] == 1) {
        OP_REQUIRES(
            ctx,
            IsValidOperand(arg1_kinds_[i], arg1_inputs_[i], num_data_inputs_,
                           num_bool_inputs_, static_cast<int>(num_steps),
                           static_cast<int>(i)) &&
                arg1_kinds_[i] == kOperandNone &&
                IsValidOperand(arg2_kinds_[i], arg2_inputs_[i],
                               num_data_inputs_, num_bool_inputs_,
                               static_cast<int>(num_steps),
                               static_cast<int>(i)) &&
                arg2_kinds_[i] == kOperandNone,
            errors::InvalidArgument("Unary step ", i,
                                    " must leave arg1/arg2 unused"));
      } else if (step_arities_[i] == 2) {
        OP_REQUIRES(
            ctx,
            IsValidOperand(arg1_kinds_[i], arg1_inputs_[i], num_data_inputs_,
                           num_bool_inputs_, static_cast<int>(num_steps),
                           static_cast<int>(i)) &&
                IsValidOperand(arg2_kinds_[i], arg2_inputs_[i],
                               num_data_inputs_, num_bool_inputs_,
                               static_cast<int>(num_steps),
                               static_cast<int>(i)) &&
                arg2_kinds_[i] == kOperandNone,
            errors::InvalidArgument("Invalid binary operand encoding at step ",
                                    i));
      } else {
        OP_REQUIRES(
            ctx,
            IsValidOperand(arg1_kinds_[i], arg1_inputs_[i], num_data_inputs_,
                           num_bool_inputs_, static_cast<int>(num_steps),
                           static_cast<int>(i)) &&
                IsValidOperand(arg2_kinds_[i], arg2_inputs_[i],
                               num_data_inputs_, num_bool_inputs_,
                               static_cast<int>(num_steps),
                               static_cast<int>(i)),
            errors::InvalidArgument("Invalid ternary operand encoding at step ",
                                    i));
        OP_REQUIRES(ctx, opcodes_[i] == kOpcodeSelect,
                    errors::InvalidArgument(
                        "Only Select may use ternary arity, step ", i));
        OP_REQUIRES(ctx, arg0_kinds_[i] == kOperandBoolInput,
                    errors::InvalidArgument(
                        "Select step ", i, " requires a bool condition"));
      }
    }
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx,
                ctx->num_inputs() == num_data_inputs_ + num_bool_inputs_,
                errors::InvalidArgument("Expected ",
                                        num_data_inputs_ + num_bool_inputs_,
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

    std::vector<TensorShape> data_input_shapes;
    data_input_shapes.reserve(num_data_inputs_);
    for (int i = 0; i < num_data_inputs_; ++i) {
      data_input_shapes.push_back(ctx->input(i).shape());
    }

    std::vector<TensorShape> bool_input_shapes;
    bool_input_shapes.reserve(num_bool_inputs_);
    for (int i = 0; i < num_bool_inputs_; ++i) {
      bool_input_shapes.push_back(ctx->input(num_data_inputs_ + i).shape());
    }

    MusaFusedElementwiseInlinePointers input_ptrs{};
    for (int i = 0; i < num_data_inputs_; ++i) {
      input_ptrs.data_ptrs[i] = ctx->input(i).tensor_data().data();
    }
    for (int i = 0; i < num_bool_inputs_; ++i) {
      input_ptrs.bool_ptrs[i] =
          ctx->input(num_data_inputs_ + i).tensor_data().data();
    }

    const MusaFusedElementwiseConfig config = BuildKernelConfig(
        data_input_shapes, bool_input_shapes, output_shape, opcodes_,
        step_arities_, arg0_kinds_, arg0_inputs_, arg1_kinds_, arg1_inputs_,
        arg2_kinds_, arg2_inputs_);

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
  int num_data_inputs_ = 0;
  int num_bool_inputs_ = 0;
  std::vector<int> opcodes_;
  std::vector<int> step_arities_;
  std::vector<int> arg0_kinds_;
  std::vector<int> arg0_inputs_;
  std::vector<int> arg1_kinds_;
  std::vector<int> arg1_inputs_;
  std::vector<int> arg2_kinds_;
  std::vector<int> arg2_inputs_;
};

#define REGISTER_MUSA_FUSED_ELEMENTWISE(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MusaFusedElementwise").Device("MUSA").TypeConstraint<TYPE>(  \
          "T"),                                                          \
      MusaFusedElementwiseOp<TYPE>);

REGISTER_MUSA_FUSED_ELEMENTWISE(float);
REGISTER_MUSA_FUSED_ELEMENTWISE(double);
REGISTER_MUSA_FUSED_ELEMENTWISE(Eigen::half);
REGISTER_MUSA_FUSED_ELEMENTWISE(bfloat16);

#undef REGISTER_MUSA_FUSED_ELEMENTWISE

}  // namespace musa
}  // namespace tensorflow
