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

#include <mudnn.h>

#include <cstdlib>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool ResolveTF32Enabled() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  if (tf32_env == nullptr) {
    return true;
  }
  return std::atoi(tf32_env) != 0;
}

}  // namespace

REGISTER_OP("MusaReshapeMatMul")
    .Input("x: T")
    .Input("w: T")
    .Output("y: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("transpose_b: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      using ::tensorflow::shape_inference::ShapeHandle;

      ShapeHandle x_shape = c->input(0);
      ShapeHandle w_shape = c->input(1);

      TF_RETURN_IF_ERROR(c->WithRankAtLeast(x_shape, 2, &x_shape));
      TF_RETURN_IF_ERROR(c->WithRank(w_shape, 2, &w_shape));

      bool transpose_b = false;
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));

      auto x_last = c->Dim(x_shape, c->Rank(x_shape) - 1);
      auto w_k = c->Dim(w_shape, transpose_b ? 1 : 0);
      auto w_n = c->Dim(w_shape, transpose_b ? 0 : 1);

      ::tensorflow::shape_inference::DimensionHandle merged_k;
      TF_RETURN_IF_ERROR(c->Merge(x_last, w_k, &merged_k));

      std::vector<::tensorflow::shape_inference::DimensionHandle> out_dims;
      out_dims.reserve(c->Rank(x_shape));
      for (int i = 0; i + 1 < c->Rank(x_shape); ++i) {
        out_dims.push_back(c->Dim(x_shape, i));
      }
      out_dims.push_back(w_n);
      c->set_output(0, c->MakeShape(out_dims));
      return Status::OK();
    });

template <typename T>
class MusaReshapeMatMulOp : public MusaOpKernel {
 public:
  explicit MusaReshapeMatMulOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    static const bool tf32_enabled_global = ResolveTF32Enabled();
    tf32_enabled_ = tf32_enabled_global;
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);

    const Tensor& x = ctx->input(0);
    const Tensor& w = ctx->input(1);

    OP_REQUIRES(ctx, x.dims() >= 2,
                errors::InvalidArgument(
                    "MusaReshapeMatMul requires x rank >= 2, got ",
                    x.shape().DebugString()));
    OP_REQUIRES(ctx, w.dims() == 2,
                errors::InvalidArgument(
                    "MusaReshapeMatMul requires 2D weight, got ",
                    w.shape().DebugString()));

    const int64_t k = x.dim_size(x.dims() - 1);
    const int64_t w_k = transpose_b_ ? w.dim_size(1) : w.dim_size(0);
    const int64_t n = transpose_b_ ? w.dim_size(0) : w.dim_size(1);

    OP_REQUIRES(ctx, k == w_k,
                errors::InvalidArgument("MusaReshapeMatMul dimension mismatch: "
                                        "x last dim=",
                                        k, ", weight K=", w_k,
                                        ", transpose_b=", transpose_b_));

    TensorShape out_shape = x.shape();
    out_shape.set_dim(x.dims() - 1, n);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    const int64_t m = x.NumElements() / k;

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    mTensor mt_x = CreateMTensor(x, format_);
    mTensor mt_w = CreateMTensor(w, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    mt_x.SetNdInfo({m, k}, {k, 1});
    mt_out.SetNdInfo({m, n}, {n, 1});

    mMatMul op;
    auto status = op.SetTranspose(false, transpose_b_);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetTranspose failed, status=",
                                 static_cast<int>(status)));

    status = op.SetAlpha(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetAlpha failed, status=",
                                 static_cast<int>(status)));

    status = op.SetBeta(0.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MatMul SetBeta failed, status=",
                                 static_cast<int>(status)));

    status = op.Run(handle, mt_out, mt_x, mt_w);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN MusaReshapeMatMul failed, status=",
                                 static_cast<int>(status)));
  }

 private:
  bool transpose_b_ = false;
  bool tf32_enabled_ = false;
};

#define REGISTER_MUSA_RESHAPE_MATMUL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MusaReshapeMatMul").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      MusaReshapeMatMulOp<TYPE>);

REGISTER_MUSA_RESHAPE_MATMUL(float);
REGISTER_MUSA_RESHAPE_MATMUL(double);
REGISTER_MUSA_RESHAPE_MATMUL(Eigen::half);
REGISTER_MUSA_RESHAPE_MATMUL(bfloat16);

#undef REGISTER_MUSA_RESHAPE_MATMUL

}  // namespace musa
}  // namespace tensorflow
