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

// ScatterNd and TensorScatterNd* op implementations using muDNN ScatterND.
//
// Supported ops:
//   - ScatterNd              : out = zeros(shape); out[indices] = updates
//   - TensorScatterUpdate    : out = tensor; out[indices] = updates
//   (UPDATE_ONLY)
//   - TensorScatterAdd       : out = tensor; out[indices] += updates (ADD)
//   - TensorScatterSub       : out = tensor; out[indices] -= updates
//                              (implemented as ADD(-updates) via
//                              Unary::SUB_BY_ALPHA,
//                               because mScatterND::Mode::SUB is unimplemented
//                               in the current mudnn runtime)
//
// muDNN ScatterND API:
//   SetMode(Mode)  : UPDATE_ONLY / ADD
//   Run(handle, self, idx, update, MemoryMaintainer)
//   where `self` is modified in-place.

#include <mudnn.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

namespace {

// Build a muDNN MemoryMaintainer backed by the TF device allocator.
::musa::dnn::MemoryMaintainer MakeMemoryMaintainer(OpKernelContext* ctx) {
  tensorflow::Allocator* tf_alloc =
      ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
  auto alloc_func =
      [tf_alloc](
          size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
    void* ptr = tf_alloc->AllocateRaw(256, size);
    return std::unique_ptr<void, std::function<void(void*)>>(
        ptr, [tf_alloc](void* p) {
          if (p) tf_alloc->DeallocateRaw(p);
        });
  };
  return ::musa::dnn::MemoryMaintainer(alloc_func);
}

// Zero-initialize a tensor via muDNN Fill.
Status ZeroInitTensor(Tensor* t, mFormat fmt, OpKernelContext* ctx) {
  if (t->NumElements() == 0) return OkStatus();
  auto& h = GetHandleByCtx(ctx);
  auto t_mt = CreateMTensor(*t, fmt);
  ::musa::dnn::Fill fill;
  auto st = fill.SetValue(0.0);
  if (st != ::musa::dnn::Status::SUCCESS)
    return errors::Internal("ScatterNd Fill SetValue failed");
  st = fill.Run(h, t_mt);
  if (st != ::musa::dnn::Status::SUCCESS)
    return errors::Internal("ScatterNd Fill Run failed");
  return OkStatus();
}

// Run muDNN ScatterND with the given mode on pre-constructed tensors.
Status RunScatterND(OpKernelContext* ctx, mTensor& self_mt,
                    const mTensor& idx_mt, const mTensor& upd_mt,
                    mScatterND::Mode mode) {
  auto& h = GetHandleByCtx(ctx);
  mScatterND scatter_nd;
  auto st = scatter_nd.SetMode(mode);
  if (st != ::musa::dnn::Status::SUCCESS)
    return errors::Internal("ScatterND SetMode failed. Status: ",
                            static_cast<int>(st));
  auto mm = MakeMemoryMaintainer(ctx);
  st = scatter_nd.Run(h, self_mt, idx_mt, upd_mt, mm);
  if (st != ::musa::dnn::Status::SUCCESS)
    return errors::Internal("ScatterND Run failed. Status: ",
                            static_cast<int>(st));
  return OkStatus();
}

}  // namespace

// ---------------------------------------------------------------------------
// ScatterNd
// Inputs: indices (0), updates (1), shape (2, HostMemory)
// Output: zeros(shape) with updates scattered at indices
// ---------------------------------------------------------------------------
template <typename T, typename IndexT>
class MusaScatterNdOp : public MusaOpKernel {
 public:
  explicit MusaScatterNdOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const Tensor& updates = ctx->input(1);
    const Tensor& shape_t = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(shape_t.shape()),
                errors::InvalidArgument(
                    "ScatterNd: shape must be a 1-D vector, got shape ",
                    shape_t.shape().DebugString()));

    // Build output shape from HostMemory shape tensor.
    TensorShape output_shape;
    auto shape_vec = shape_t.flat<IndexT>();
    for (int i = 0; i < shape_t.NumElements(); ++i) {
      OP_REQUIRES(ctx, shape_vec(i) >= 0,
                  errors::InvalidArgument(
                      "ScatterNd: shape values must be non-negative, got ",
                      shape_vec(i), " at index ", i));
      output_shape.AddDim(static_cast<int64_t>(shape_vec(i)));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Zero-initialize output.
    OP_REQUIRES_OK(ctx, ZeroInitTensor(output, format_, ctx));

    if (indices.NumElements() == 0 || updates.NumElements() == 0) return;

    mTensor out_mt = CreateMTensor(*output, format_);
    mTensor idx_mt = CreateMTensor(indices, format_);
    mTensor upd_mt = CreateMTensor(updates, format_);

    OP_REQUIRES_OK(ctx, RunScatterND(ctx, out_mt, idx_mt, upd_mt,
                                     mScatterND::Mode::UPDATE_ONLY));
  }
};

// ---------------------------------------------------------------------------
// TensorScatterNd base: copy input tensor to output, then scatter.
// Inputs: tensor (0), indices (1), updates (2)
// Output: modified copy of tensor
// ---------------------------------------------------------------------------
template <typename T, typename IndexT>
class MusaTensorScatterNdOp : public MusaOpKernel {
 public:
  explicit MusaTensorScatterNdOp(OpKernelConstruction* ctx,
                                 mScatterND::Mode mode)
      : MusaOpKernel(ctx), mode_(mode) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output));

    // Copy input tensor to output (device-to-device).
    if (tensor.NumElements() > 0) {
      auto stream = GetMusaStreamByCtx(ctx);
      musaError_t err =
          musaMemcpyAsync(output->data(), tensor.data(),
                          static_cast<size_t>(tensor.NumElements()) *
                              DataTypeSize(tensor.dtype()),
                          musaMemcpyDeviceToDevice, stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("TensorScatterNd musaMemcpyAsync failed: ",
                                   musaGetErrorString(err)));
    }

    if (indices.NumElements() == 0 || updates.NumElements() == 0) return;

    mTensor out_mt = CreateMTensor(*output, format_);
    mTensor idx_mt = CreateMTensor(indices, format_);
    mTensor upd_mt = CreateMTensor(updates, format_);

    OP_REQUIRES_OK(ctx, RunScatterND(ctx, out_mt, idx_mt, upd_mt, mode_));
  }

 private:
  mScatterND::Mode mode_;
};

// Concrete subclasses for each TensorScatter variant.
template <typename T, typename IndexT>
class MusaTensorScatterUpdateOp : public MusaTensorScatterNdOp<T, IndexT> {
 public:
  explicit MusaTensorScatterUpdateOp(OpKernelConstruction* ctx)
      : MusaTensorScatterNdOp<T, IndexT>(ctx, mScatterND::Mode::UPDATE_ONLY) {}
};

template <typename T, typename IndexT>
class MusaTensorScatterAddOp : public MusaTensorScatterNdOp<T, IndexT> {
 public:
  explicit MusaTensorScatterAddOp(OpKernelConstruction* ctx)
      : MusaTensorScatterNdOp<T, IndexT>(ctx, mScatterND::Mode::ADD) {}
};

// TensorScatterSub: implemented as ADD(-updates) because mScatterND::Mode::SUB
// is not yet supported at runtime in the current mudnn version.
template <typename T, typename IndexT>
class MusaTensorScatterSubOp : public MusaOpKernel {
 public:
  explicit MusaTensorScatterSubOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output));

    // Copy input tensor to output (device-to-device).
    if (tensor.NumElements() > 0) {
      auto stream = GetMusaStreamByCtx(ctx);
      musaError_t err =
          musaMemcpyAsync(output->data(), tensor.data(),
                          static_cast<size_t>(tensor.NumElements()) *
                              DataTypeSize(tensor.dtype()),
                          musaMemcpyDeviceToDevice, stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("TensorScatterSub musaMemcpyAsync failed: ",
                                   musaGetErrorString(err)));
    }

    if (indices.NumElements() == 0 || updates.NumElements() == 0) return;

    // Negate updates via Unary::SUB_BY_ALPHA(alpha=0): out = 0 - updates =
    // -updates.
    Tensor neg_updates;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(updates.dtype(), updates.shape(),
                                           &neg_updates));

    auto& h = GetHandleByCtx(ctx);
    mUnary neg_op;
    neg_op.SetMode(::musa::dnn::Unary::Mode::SUB_BY_ALPHA);
    neg_op.SetAlpha(0.0);
    mTensor upd_mt = CreateMTensor(updates, format_);
    mTensor neg_upd_mt = CreateMTensor(neg_updates, format_);
    auto st = neg_op.Run(h, neg_upd_mt, upd_mt);
    OP_REQUIRES(ctx, st == ::musa::dnn::Status::SUCCESS,
                errors::Internal("TensorScatterSub negate failed. Status: ",
                                 static_cast<int>(st)));

    // Scatter-add negated updates into output.
    mTensor out_mt = CreateMTensor(*output, format_);
    mTensor idx_mt = CreateMTensor(indices, format_);
    OP_REQUIRES_OK(ctx, RunScatterND(ctx, out_mt, idx_mt, neg_upd_mt,
                                     mScatterND::Mode::ADD));
  }
};

// ---------------------------------------------------------------------------
// Kernel registrations
// ---------------------------------------------------------------------------

// ScatterNd: shape is provided as HostMemory (CPU tensor)
#define REGISTER_SCATTER_ND(T, IndexT)                            \
  REGISTER_KERNEL_BUILDER(Name("ScatterNd")                       \
                              .Device(DEVICE_MTGPU)               \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<IndexT>("Tindices") \
                              .HostMemory("shape"),               \
                          MusaScatterNdOp<T, IndexT>);

#define REGISTER_SCATTER_ND_INDEX(T) \
  REGISTER_SCATTER_ND(T, int32)      \
  REGISTER_SCATTER_ND(T, int64)

REGISTER_SCATTER_ND_INDEX(float)
REGISTER_SCATTER_ND_INDEX(double)
REGISTER_SCATTER_ND_INDEX(int32)
REGISTER_SCATTER_ND_INDEX(int64)
REGISTER_SCATTER_ND_INDEX(Eigen::half)
REGISTER_SCATTER_ND_INDEX(bfloat16)

#undef REGISTER_SCATTER_ND_INDEX
#undef REGISTER_SCATTER_ND

// TensorScatterUpdate
#define REGISTER_TENSOR_SCATTER_UPDATE(T, IndexT)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterUpdate")              \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<T>("T")              \
                              .TypeConstraint<IndexT>("Tindices"), \
                          MusaTensorScatterUpdateOp<T, IndexT>);

#define REGISTER_TENSOR_SCATTER_UPDATE_INDEX(T) \
  REGISTER_TENSOR_SCATTER_UPDATE(T, int32)      \
  REGISTER_TENSOR_SCATTER_UPDATE(T, int64)

REGISTER_TENSOR_SCATTER_UPDATE_INDEX(float)
REGISTER_TENSOR_SCATTER_UPDATE_INDEX(double)
REGISTER_TENSOR_SCATTER_UPDATE_INDEX(int32)
REGISTER_TENSOR_SCATTER_UPDATE_INDEX(int64)
REGISTER_TENSOR_SCATTER_UPDATE_INDEX(Eigen::half)
REGISTER_TENSOR_SCATTER_UPDATE_INDEX(bfloat16)

#undef REGISTER_TENSOR_SCATTER_UPDATE_INDEX
#undef REGISTER_TENSOR_SCATTER_UPDATE

// TensorScatterAdd
#define REGISTER_TENSOR_SCATTER_ADD(T, IndexT)                     \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterAdd")                 \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<T>("T")              \
                              .TypeConstraint<IndexT>("Tindices"), \
                          MusaTensorScatterAddOp<T, IndexT>);

#define REGISTER_TENSOR_SCATTER_ADD_INDEX(T) \
  REGISTER_TENSOR_SCATTER_ADD(T, int32)      \
  REGISTER_TENSOR_SCATTER_ADD(T, int64)

REGISTER_TENSOR_SCATTER_ADD_INDEX(float)
REGISTER_TENSOR_SCATTER_ADD_INDEX(double)
REGISTER_TENSOR_SCATTER_ADD_INDEX(int32)
REGISTER_TENSOR_SCATTER_ADD_INDEX(int64)
REGISTER_TENSOR_SCATTER_ADD_INDEX(Eigen::half)
REGISTER_TENSOR_SCATTER_ADD_INDEX(bfloat16)

#undef REGISTER_TENSOR_SCATTER_ADD_INDEX
#undef REGISTER_TENSOR_SCATTER_ADD

// TensorScatterSub
#define REGISTER_TENSOR_SCATTER_SUB(T, IndexT)                     \
  REGISTER_KERNEL_BUILDER(Name("TensorScatterSub")                 \
                              .Device(DEVICE_MTGPU)                \
                              .TypeConstraint<T>("T")              \
                              .TypeConstraint<IndexT>("Tindices"), \
                          MusaTensorScatterSubOp<T, IndexT>);

#define REGISTER_TENSOR_SCATTER_SUB_INDEX(T) \
  REGISTER_TENSOR_SCATTER_SUB(T, int32)      \
  REGISTER_TENSOR_SCATTER_SUB(T, int64)

REGISTER_TENSOR_SCATTER_SUB_INDEX(float)
REGISTER_TENSOR_SCATTER_SUB_INDEX(double)
REGISTER_TENSOR_SCATTER_SUB_INDEX(int32)
REGISTER_TENSOR_SCATTER_SUB_INDEX(int64)
REGISTER_TENSOR_SCATTER_SUB_INDEX(Eigen::half)
REGISTER_TENSOR_SCATTER_SUB_INDEX(bfloat16)

#undef REGISTER_TENSOR_SCATTER_SUB_INDEX
#undef REGISTER_TENSOR_SCATTER_SUB

}  // namespace musa
}  // namespace tensorflow
