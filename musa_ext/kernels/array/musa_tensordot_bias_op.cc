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

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "../utils_op.h"
#include "musa_transpose_functor.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

// =============================================================================
// Op Registration - 必须在 namespace 内部
// =============================================================================

REGISTER_OP("MusaTensorDotBias")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("axes_a: list(int)")
    .Attr("axes_b: list(int)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle a_shape = c->input(0);
      shape_inference::ShapeHandle b_shape = c->input(1);
      shape_inference::ShapeHandle bias_shape = c->input(2);

      if (!c->RankKnown(a_shape) || !c->RankKnown(b_shape)) {
        c->set_output(0, c->UnknownShape());
        return ::tsl::OkStatus();
      }

      std::vector<int> axes_a, axes_b;
      TF_RETURN_IF_ERROR(c->GetAttr("axes_a", &axes_a));
      TF_RETURN_IF_ERROR(c->GetAttr("axes_b", &axes_b));

      int a_rank = c->Rank(a_shape);
      int b_rank = c->Rank(b_shape);

      std::vector<shape_inference::DimensionHandle> output_dims;

      // A 的非收缩维度
      std::unordered_set<int> a_contract_set;
      for (int ax : axes_a) {
        int norm = ax < 0 ? ax + a_rank : ax;
        a_contract_set.insert(norm);
      }
      for (int i = 0; i < a_rank; ++i) {
        if (a_contract_set.find(i) == a_contract_set.end()) {
          output_dims.push_back(c->Dim(a_shape, i));
        }
      }

      // B 的非收缩维度
      std::unordered_set<int> b_contract_set;
      for (int ax : axes_b) {
        int norm = ax < 0 ? ax + b_rank : ax;
        b_contract_set.insert(norm);
      }
      for (int i = 0; i < b_rank; ++i) {
        if (b_contract_set.find(i) == b_contract_set.end()) {
          output_dims.push_back(c->Dim(b_shape, i));
        }
      }

      c->set_output(0, c->MakeShape(output_dims));
      return ::tsl::OkStatus();
    });

// =============================================================================
// TensorDot 维度计算辅助结构和函数
// =============================================================================

namespace {

struct TensorDotDims {
  int64_t a_batch_size;
  int64_t a_contract_size;
  int64_t b_contract_size;
  int64_t b_batch_size;

  std::vector<int64_t> output_dims;
  std::vector<int64_t> a_perm;
  std::vector<int64_t> b_perm;
};

Status ComputeTensorDotDims(const TensorShape& a_shape,
                            const TensorShape& b_shape,
                            const std::vector<int>& axes_a,
                            const std::vector<int>& axes_b,
                            TensorDotDims* dims) {
  const int a_rank = a_shape.dims();
  const int b_rank = b_shape.dims();

  if (axes_a.size() != axes_b.size()) {
    return errors::InvalidArgument(
        "axes_a and axes_b must have the same size. Got ", axes_a.size(),
        " and ", axes_b.size());
  }

  std::vector<int> norm_axes_a(axes_a.size());
  std::vector<int> norm_axes_b(axes_b.size());

  for (size_t i = 0; i < axes_a.size(); ++i) {
    norm_axes_a[i] = axes_a[i] < 0 ? axes_a[i] + a_rank : axes_a[i];
    norm_axes_b[i] = axes_b[i] < 0 ? axes_b[i] + b_rank : axes_b[i];

    if (norm_axes_a[i] < 0 || norm_axes_a[i] >= a_rank) {
      return errors::InvalidArgument("axes_a[", i, "]=", axes_a[i],
                                     " out of bounds for rank ", a_rank);
    }
    if (norm_axes_b[i] < 0 || norm_axes_b[i] >= b_rank) {
      return errors::InvalidArgument("axes_b[", i, "]=", axes_b[i],
                                     " out of bounds for rank ", b_rank);
    }

    if (a_shape.dim_size(norm_axes_a[i]) != b_shape.dim_size(norm_axes_b[i])) {
      return errors::InvalidArgument(
          "Contraction dimensions must match. A[", norm_axes_a[i],
          "]=", a_shape.dim_size(norm_axes_a[i]), " vs B[", norm_axes_b[i],
          "]=", b_shape.dim_size(norm_axes_b[i]));
    }
  }

  std::unordered_set<int> a_contract_set(norm_axes_a.begin(),
                                         norm_axes_a.end());
  std::unordered_set<int> b_contract_set(norm_axes_b.begin(),
                                         norm_axes_b.end());

  // A: 非收缩轴在前，收缩轴在后 => [batch, contract]
  dims->a_perm.clear();
  dims->a_batch_size = 1;
  for (int i = 0; i < a_rank; ++i) {
    if (a_contract_set.find(i) == a_contract_set.end()) {
      dims->a_perm.push_back(i);
      dims->a_batch_size *= a_shape.dim_size(i);
    }
  }
  dims->a_contract_size = 1;
  for (int axis : norm_axes_a) {
    dims->a_perm.push_back(axis);
    dims->a_contract_size *= a_shape.dim_size(axis);
  }

  // B: 收缩轴在前，非收缩轴在后 => [contract, batch]
  dims->b_perm.clear();
  dims->b_contract_size = 1;
  for (int axis : norm_axes_b) {
    dims->b_perm.push_back(axis);
    dims->b_contract_size *= b_shape.dim_size(axis);
  }
  dims->b_batch_size = 1;
  for (int i = 0; i < b_rank; ++i) {
    if (b_contract_set.find(i) == b_contract_set.end()) {
      dims->b_perm.push_back(i);
      dims->b_batch_size *= b_shape.dim_size(i);
    }
  }

  // 输出维度
  dims->output_dims.clear();
  for (int i = 0; i < a_rank; ++i) {
    if (a_contract_set.find(i) == a_contract_set.end()) {
      dims->output_dims.push_back(a_shape.dim_size(i));
    }
  }
  for (int i = 0; i < b_rank; ++i) {
    if (b_contract_set.find(i) == b_contract_set.end()) {
      dims->output_dims.push_back(b_shape.dim_size(i));
    }
  }

  return ::tsl::OkStatus();
}

template <typename T>
bool IsIdentityPermutation(const std::vector<T>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<T>(i)) {
      return false;
    }
  }
  return true;
}

}  // namespace

// =============================================================================
// MusaTensorDotBiasOp Implementation
// =============================================================================

template <typename T>
class MusaTensorDotBiasOp : public MusaOpKernel {
 public:
  explicit MusaTensorDotBiasOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_a", &axes_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axes_b", &axes_b_));

    // TF32 加速控制
    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return true;  // 默认开启
    }();
    tf32_enabled_ = tf32_enabled_global;
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    VLOG(2) << "MusaTensorDotBias: A shape = " << a.shape().DebugString()
            << ", B shape = " << b.shape().DebugString()
            << ", bias shape = " << bias.shape().DebugString();

    // 计算输出形状
    TensorDotDims dims;
    OP_REQUIRES_OK(ctx, ComputeTensorDotDims(a.shape(), b.shape(), axes_a_,
                                             axes_b_, &dims));

    TensorShape output_shape;
    for (int64_t dim : dims.output_dims) {
      output_shape.AddDim(dim);
    }

    // 验证 bias 形状
    // mMatMul::RunWithBiasAdd 期望 bias 是 1D 向量，大小为 N (b_batch_size)
    // 或者是可以 reshape 为 1D 的形状
    int64_t n_dim = dims.b_batch_size;

    if (bias.NumElements() != n_dim && bias.NumElements() != 1) {
      // bias 元素数不匹配，检查是否可以 broadcast
      OP_REQUIRES(ctx,
          bias.dims() == 2 && bias.dim_size(1) == n_dim,
          errors::InvalidArgument(
              "Bias shape ", bias.shape().DebugString(),
              " is incompatible with output width N=", n_dim,
              ". Expected 1D vector of size ", n_dim, " or shape compatible with it"));
    }

    VLOG(2) << "MusaTensorDotBias: bias shape validated, n_dim=" << n_dim
            << ", bias elements=" << bias.NumElements();

    // 分配输出
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // 空张量快速返回
    if (output->NumElements() == 0) {
      return;
    }

    // 设置 TF32
    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    // 执行计算
    OP_REQUIRES_OK(ctx, DoTensorDotBias(ctx, a, b, bias, dims, output));

    VLOG(2) << "MusaTensorDotBias: output shape = "
            << output->shape().DebugString();
  }

 private:
  std::vector<int> axes_a_;
  std::vector<int> axes_b_;
  bool tf32_enabled_;

  // 执行 Transpose 操作
  Status DoTranspose(OpKernelContext* ctx, const Tensor& input,
                     const std::vector<int64_t>& perm, Tensor* output) {
    mTensor input_mt = CreateMTensor(input);
    mTensor output_mt = CreateMTensor(*output);
    return TransposeFunctor::Compute(ctx, input_mt, perm, output_mt);
  }

  // 执行带 Bias 的 MatMul 操作
  Status DoMatMulWithBias(OpKernelContext* ctx, const Tensor& a,
                          const Tensor& b, const Tensor& bias, Tensor* output) {
    mHandle& handle = GetHandleByCtx(ctx);

    mTensor mt_a = CreateMTensor(a);
    mTensor mt_b = CreateMTensor(b);
    mTensor mt_bias = CreateMTensor(bias);
    mTensor mt_out = CreateMTensor(*output);

    mMatMul op;
    op.SetTranspose(false, false);
    op.SetAlpha(1.0);
    op.SetBeta(0.0);

    // 使用 MemoryMaintainer
    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      auto deleter = [tf_allocator](void* p) {
        if (p) tf_allocator->DeallocateRaw(p);
      };
      return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };
    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    // 使用 RunWithBiasAdd
    auto status = op.RunWithBiasAdd(handle, mt_out, mt_a, mt_b, mt_bias, mm);

    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA TensorDotBias MatMul with bias failed. Status: ",
                              static_cast<int>(status));
    }

    return ::tsl::OkStatus();
  }

  Status DoTensorDotBias(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                         const Tensor& bias, const TensorDotDims& dims,
                         Tensor* output) {
    bool need_transpose_a = !IsIdentityPermutation(dims.a_perm);
    bool need_transpose_b = !IsIdentityPermutation(dims.b_perm);

    VLOG(2) << "MusaTensorDotBias: need_transpose_a=" << need_transpose_a
            << ", need_transpose_b=" << need_transpose_b
            << ", a_batch=" << dims.a_batch_size
            << ", a_contract=" << dims.a_contract_size
            << ", b_contract=" << dims.b_contract_size
            << ", b_batch=" << dims.b_batch_size;

    Tensor a_2d;
    TF_RETURN_IF_ERROR(PrepareTensor(ctx, a, dims.a_perm, dims.a_batch_size,
                                     dims.a_contract_size, need_transpose_a,
                                     &a_2d));

    Tensor b_2d;
    TF_RETURN_IF_ERROR(PrepareTensor(ctx, b, dims.b_perm, dims.b_contract_size,
                                     dims.b_batch_size, need_transpose_b,
                                     &b_2d));

    // 处理 bias：如果需要，reshape 为 2D
    Tensor bias_2d;
    TF_RETURN_IF_ERROR(PrepareBiasTensor(ctx, bias, dims, &bias_2d));

    // [a_batch, a_contract] x [b_contract, b_batch] = [a_batch, b_batch]
    TensorShape matmul_shape({dims.a_batch_size, dims.b_batch_size});

    // output 的 buffer 已经分配好了，只需要用 2D view 去写入
    Tensor matmul_view;
    if (!matmul_view.CopyFrom(*output, matmul_shape)) {
      Tensor matmul_temp;
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(a.dtype(), matmul_shape, &matmul_temp));
      TF_RETURN_IF_ERROR(DoMatMulWithBias(ctx, a_2d, b_2d, bias_2d, &matmul_temp));

      TensorShape output_shape;
      for (int64_t dim : dims.output_dims) output_shape.AddDim(dim);
      if (!output->CopyFrom(matmul_temp, output_shape)) {
        return errors::Internal("Failed to reshape matmul result to output");
      }
      return ::tsl::OkStatus();
    }

    TF_RETURN_IF_ERROR(DoMatMulWithBias(ctx, a_2d, b_2d, bias_2d, &matmul_view));
    return ::tsl::OkStatus();
  }

  // 准备张量：transpose + reshape 为 2D
  Status PrepareTensor(OpKernelContext* ctx, const Tensor& input,
                       const std::vector<int64_t>& perm, int64_t dim0,
                       int64_t dim1, bool need_transpose, Tensor* output) {
    TensorShape target_shape({dim0, dim1});

    if (need_transpose) {
      // 计算 transpose 后的形状
      TensorShape transposed_shape;
      for (size_t i = 0; i < perm.size(); ++i) {
        transposed_shape.AddDim(input.dim_size(perm[i]));
      }

      Tensor transposed;
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(input.dtype(), transposed_shape, &transposed));

      // 执行 transpose
      TF_RETURN_IF_ERROR(DoTranspose(ctx, input, perm, &transposed));

      // Reshape (零开销，只改变视图)
      if (!output->CopyFrom(transposed, target_shape)) {
        return errors::Internal(
            "Reshape after transpose failed. transposed shape: ",
            transposed.shape().DebugString(),
            ", target shape: ", target_shape.DebugString());
      }
    } else {
      // 直接 reshape
      if (!output->CopyFrom(input, target_shape)) {
        return errors::Internal(
            "Reshape failed. input shape: ", input.shape().DebugString(),
            ", target shape: ", target_shape.DebugString());
      }
    }

    return ::tsl::OkStatus();
  }

  // 准备 bias 张量：reshape 为 1D 向量 (N 维度)
  Status PrepareBiasTensor(OpKernelContext* ctx, const Tensor& bias,
                           const TensorDotDims& dims, Tensor* output) {
    // mMatMul::RunWithBiasAdd 期望 bias 是 1D 向量，大小为 N (b_batch_size)
    int64_t n_dim = dims.b_batch_size;

    // 如果 bias 已经是正确的 1D 形状
    if (bias.dims() == 1 && bias.dim_size(0) == n_dim) {
      if (!output->CopyFrom(bias, bias.shape())) {
        return errors::Internal("Failed to copy bias");
      }
      return ::tsl::OkStatus();
    }

    // 如果 bias 是标量，需要 broadcast 到向量
    if (bias.NumElements() == 1) {
      TensorShape target_shape({n_dim});
      TF_RETURN_IF_ERROR(ctx->allocate_temp(bias.dtype(), target_shape, output));
      // TODO: 实现标量到向量的 broadcast
      return ::tsl::OkStatus();
    }

    // 如果 bias 是 2D [M, N]，需要提取 N 维度作为向量
    if (bias.dims() == 2 && bias.dim_size(1) == n_dim) {
      // 假设 bias 沿 M 维度是相同的，取第一行作为向量
      TensorShape target_shape({n_dim});
      TF_RETURN_IF_ERROR(ctx->allocate_temp(bias.dtype(), target_shape, output));
      // TODO: 实现从 2D 提取 1D 的逻辑
      return ::tsl::OkStatus();
    }

    // 尝试直接 reshape
    TensorShape target_shape({n_dim});
    if (!output->CopyFrom(bias, target_shape)) {
      return errors::Internal(
          "Failed to reshape bias from ", bias.shape().DebugString(),
          " to ", target_shape.DebugString());
    }

    return ::tsl::OkStatus();
  }
};

// =============================================================================
// Kernel Registration
// =============================================================================

#define REGISTER_MUSA_TENSORDOT_BIAS(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("MusaTensorDotBias").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaTensorDotBiasOp<TYPE>);

REGISTER_MUSA_TENSORDOT_BIAS(float);
REGISTER_MUSA_TENSORDOT_BIAS(double);
REGISTER_MUSA_TENSORDOT_BIAS(Eigen::half);
REGISTER_MUSA_TENSORDOT_BIAS(bfloat16);

#undef REGISTER_MUSA_TENSORDOT_BIAS

}  // namespace musa
}  // namespace tensorflow
