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

#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

Status ConfigureLastDimBroadcastView(const Tensor& input, const Tensor& tensor,
                                     mTensor* mt, const char* input_name) {
  if (tensor.shape() == input.shape()) {
    return Status::OK();
  }

  if (input.dims() < 1) {
    return errors::InvalidArgument(
        "MusaShiftedAffineMap expects input rank >= 1");
  }

  const int last_dim = input.dims() - 1;
  if (tensor.dims() == 1 && tensor.dim_size(0) == input.dim_size(last_dim)) {
    std::vector<int64_t> dims(input.dims(), 1);
    std::vector<int64_t> strides(input.dims(), 0);
    for (int i = 0; i < input.dims(); ++i) {
      dims[i] = input.dim_size(i);
    }
    dims[last_dim] = tensor.dim_size(0);
    strides[last_dim] = 1;

    auto status = mt->SetNdInfo(input.dims(), dims.data(), strides.data());
    if (status == ::musa::dnn::Status::SUCCESS) {
      return Status::OK();
    }
    return errors::Internal(
        "MusaShiftedAffineMap SetNdInfo failed for ", input_name,
        ". Status: ", static_cast<int>(status));
  }

  return errors::InvalidArgument(
      "MusaShiftedAffineMap expects ", input_name,
      " to match input shape or be rank-1 with size equal to input last ",
      "dimension. Got ", tensor.shape().DebugString(), " vs input ",
      input.shape().DebugString());
}

}  // namespace

template <typename T>
class MusaShiftedAffineMapOp : public MusaOpKernel {
 public:
  explicit MusaShiftedAffineMapOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    Tensor* output = nullptr;
    const std::vector<int> forwardable_input_indices = {0};
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 forwardable_input_indices, 0, input.shape(), &output));

    if (input.NumElements() == 0 || scale.NumElements() == 0 ||
        bias.NumElements() == 0) {
      return;
    }

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_scale = CreateMTensor(scale, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    OP_REQUIRES_OK(ctx, ConfigureLastDimBroadcastView(input, scale, &mt_scale,
                                                      "scale"));
    OP_REQUIRES_OK(ctx, ConfigureLastDimBroadcastView(input, bias, &mt_bias,
                                                      "bias"));

    mTernary ternary_op;
    auto mode_status = ternary_op.SetMode(::musa::dnn::Ternary::Mode::ADDCMUL);
    OP_REQUIRES(ctx, mode_status == mStatus::SUCCESS,
                errors::Internal(
                    "MUSA MusaShiftedAffineMap SetMode(ADDCMUL) failed. ",
                    "Status: ", static_cast<int>(mode_status)));

    auto run_status = ternary_op.Run(handle, mt_output, mt_bias, mt_input,
                                     mt_scale);
    OP_REQUIRES(ctx, run_status == mStatus::SUCCESS,
                errors::Internal(
                    "MUSA MusaShiftedAffineMap execution failed. Status: ",
                    static_cast<int>(run_status)));
  }
};

}  // namespace musa

REGISTER_OP("MusaShiftedAffineMap")
    .Input("input: T")
    .Input("scale: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow

REGISTER_KERNEL_BUILDER(
    Name("MusaShiftedAffineMap").Device(DEVICE_MTGPU).TypeConstraint<float>("T"),
    ::tensorflow::musa::MusaShiftedAffineMapOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MusaShiftedAffineMap").Device(DEVICE_MTGPU).TypeConstraint<double>("T"),
    ::tensorflow::musa::MusaShiftedAffineMapOp<double>);
REGISTER_KERNEL_BUILDER(
    Name("MusaShiftedAffineMap").Device(DEVICE_MTGPU).TypeConstraint<Eigen::half>("T"),
    ::tensorflow::musa::MusaShiftedAffineMapOp<Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("MusaShiftedAffineMap").Device(DEVICE_MTGPU).TypeConstraint<::tensorflow::bfloat16>("T"),
    ::tensorflow::musa::MusaShiftedAffineMapOp<::tensorflow::bfloat16>);
