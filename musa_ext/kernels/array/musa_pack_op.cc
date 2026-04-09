// MUSA Pack/Unpack Operators
// Pack: implemented via muDNN Concat with expanded dimension metadata
// Unpack: implemented via custom kernel for direct memory copy
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.

#include <mudnn.h>

#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

template <typename T>
inline bool NeedsHostVisiblePackSync() {
  return std::is_same<T, int32>::value || std::is_same<T, int64>::value;
}

inline void SyncPackStreamIfNeeded(OpKernelContext* ctx, musaStream_t stream,
                                   bool should_sync) {
  if (!should_sync) return;

  musaError_t err = musaStreamSynchronize(stream);
  OP_REQUIRES(ctx, err == musaSuccess,
              errors::Internal("MUSA Pack stream sync failed: ",
                               musaGetErrorString(err)));
}

// Create mTensor view with an extra dimension of size 1 at specified axis.
mTensor CreateMTensorWithExpandedDim(const Tensor& t, int axis,
                                     mFormat format) {
  mTensor rst = CreateMTensor(t, format);

  auto orig_dims = t.shape().dim_sizes();
  const int orig_rank = static_cast<int>(orig_dims.size());
  std::vector<int64_t> new_dims(orig_rank + 1);
  for (int i = 0; i < axis; ++i) new_dims[i] = orig_dims[i];
  new_dims[axis] = 1;
  for (int i = axis; i < orig_rank; ++i) new_dims[i + 1] = orig_dims[i];

  rst.SetNdInfo(orig_rank + 1, new_dims.data());
  return rst;
}

// Pack concatenates tensors along a new dimension via muDNN Concat.
template <typename T>
class MusaPackOp : public MusaOpKernel {
 public:
  explicit MusaPackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const int N = ctx->num_inputs();

    OP_REQUIRES(ctx, N > 0,
                errors::InvalidArgument("Pack requires at least one input"));

    // Get shapes from all inputs
    std::vector<TensorShape> shapes(N);
    for (int i = 0; i < N; ++i) {
      shapes[i] = ctx->input(i).shape();
    }

    // Verify all shapes match
    for (int i = 1; i < N; ++i) {
      OP_REQUIRES(ctx, shapes[i].IsSameSize(shapes[0]),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: input 0 has shape ",
                      shapes[0].DebugString(), " but input ", i, " has shape ",
                      shapes[i].DebugString()));
    }

    // Compute output shape
    const int dims = shapes[0].dims();
    int axis = axis_ < 0 ? axis_ + dims + 1 : axis_;
    OP_REQUIRES(ctx, axis >= 0 && axis <= dims,
                errors::InvalidArgument("axis must be in range [", -dims - 1,
                                        ", ", dims, "], but got ", axis_));

    TensorShape output_shape;
    for (int i = 0; i < axis; ++i) {
      output_shape.AddDim(shapes[0].dim_size(i));
    }
    output_shape.AddDim(N);
    for (int i = axis; i < dims; ++i) {
      output_shape.AddDim(shapes[0].dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Handle empty tensors
    if (output->NumElements() == 0) return;

    // Handle single input - just copy with expanded dim
    if (N == 1) {
      musaStream_t stream = GetMusaStreamByCtx(ctx);
      musaError_t err = musaMemcpyAsync(
          const_cast<char*>(output->tensor_data().data()),
          ctx->input(0).tensor_data().data(), ctx->input(0).TotalBytes(),
          musaMemcpyDeviceToDevice, stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("musaMemcpyAsync failed: ",
                                   musaGetErrorString(err)));
      SyncPackStreamIfNeeded(ctx, stream, NeedsHostVisiblePackSync<T>());
      return;
    }

    // Use muDNN Concat with expanded dimension metadata
    auto& handle = GetHandleByCtx(ctx);

    // Create mTensor views with expanded dimension at axis
    std::vector<::musa::dnn::Tensor> mudnn_ins;
    mudnn_ins.reserve(N);
    for (int i = 0; i < N; ++i) {
      mudnn_ins.push_back(
          CreateMTensorWithExpandedDim(ctx->input(i), axis, format_));
    }

    ::musa::dnn::Tensor mudnn_out = CreateMTensor(*output, format_);
    ::musa::dnn::Concat concat_op;
    concat_op.SetAxis(axis);

    auto status = concat_op.Run(handle, mudnn_out, N, mudnn_ins.data());

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Concat Run failed for Pack. Status: ",
                                 static_cast<int>(status)));
    SyncPackStreamIfNeeded(ctx, reinterpret_cast<musaStream_t>(handle.GetStream()),
                           NeedsHostVisiblePackSync<T>());
  }

 private:
  int axis_;
};

// Unpack kernel launchers (implemented in musa_pack_unpack_kernel.mu)
extern "C" {
void LaunchUnpackSingleFloat(const float* input, float* output,
                             int64_t outer_size, int64_t N, int64_t inner_size,
                             int64_t unpack_idx, musaStream_t stream);
void LaunchUnpackSingleDouble(const double* input, double* output,
                              int64_t outer_size, int64_t N, int64_t inner_size,
                              int64_t unpack_idx, musaStream_t stream);
void LaunchUnpackSingleInt32(const int32_t* input, int32_t* output,
                             int64_t outer_size, int64_t N, int64_t inner_size,
                             int64_t unpack_idx, musaStream_t stream);
void LaunchUnpackSingleInt64(const int64_t* input, int64_t* output,
                             int64_t outer_size, int64_t N, int64_t inner_size,
                             int64_t unpack_idx, musaStream_t stream);
void LaunchUnpackSingleUInt8(const uint8_t* input, uint8_t* output,
                             int64_t outer_size, int64_t N, int64_t inner_size,
                             int64_t unpack_idx, musaStream_t stream);
void LaunchUnpackSingleBool(const bool* input, bool* output, int64_t outer_size,
                            int64_t N, int64_t inner_size, int64_t unpack_idx,
                            musaStream_t stream);
void LaunchUnpackSingleHalf(const void* input, void* output, int64_t outer_size,
                            int64_t N, int64_t inner_size, int64_t unpack_idx,
                            musaStream_t stream);
void LaunchUnpackSingleBFloat16(const void* input, void* output,
                                int64_t outer_size, int64_t N,
                                int64_t inner_size, int64_t unpack_idx,
                                musaStream_t stream);

}  // extern "C"

// Unpack splits a tensor along a dimension into multiple tensors.
template <typename T>
class MusaUnpackOp : public MusaOpKernel {
 public:
  explicit MusaUnpackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num", &num_outputs_attr_));
  }

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    MUSA_DEBUG_LOG_KERNEL(ctx);
    const Tensor& input = ctx->input(0);
    const int N = num_outputs_attr_;

    const int dims = input.dims();
    int axis = axis_ < 0 ? axis_ + dims : axis_;

    OP_REQUIRES(ctx, axis >= 0 && axis < dims,
                errors::InvalidArgument("axis must be in range [", -dims, ", ",
                                        dims, "), but got ", axis_));

    OP_REQUIRES(ctx, N == input.dim_size(axis),
                errors::InvalidArgument("num outputs (", N,
                                        ") must equal the dimension on axis (",
                                        input.dim_size(axis), ")"));

    // Compute output shape (input shape without the axis dimension)
    TensorShape output_shape;
    for (int i = 0; i < dims; ++i) {
      if (i != axis) {
        output_shape.AddDim(input.dim_size(i));
      }
    }

    // Allocate outputs
    std::vector<Tensor*> outputs(N);
    for (int i = 0; i < N; ++i) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
    }

    // Handle empty tensors
    if (input.NumElements() == 0) return;

    // Handle single output - just copy
    if (N == 1) {
      musaStream_t stream = GetMusaStreamByCtx(ctx);
      musaError_t err =
          musaMemcpyAsync(const_cast<char*>(outputs[0]->tensor_data().data()),
                          input.tensor_data().data(), input.TotalBytes(),
                          musaMemcpyDeviceToDevice, stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("musaMemcpyAsync failed: ",
                                   musaGetErrorString(err)));
      return;
    }

    // Compute sizes for kernel
    // outer_size: product of dimensions before axis
    // inner_size: product of dimensions after axis
    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
      outer_size *= input.dim_size(i);
    }
    int64_t inner_size = 1;
    for (int i = axis + 1; i < dims; ++i) {
      inner_size *= input.dim_size(i);
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    // Fast path for axis=0: use async memcpy for contiguous copies
    // Input layout [N, d0, d1, ...] means each output is a contiguous slice
    if (axis == 0) {
      const int64_t output_bytes = outputs[0]->TotalBytes();
      for (int i = 0; i < N; ++i) {
        musaError_t err = musaMemcpyAsync(
            outputs[i]->flat<T>().data(),
            reinterpret_cast<const char*>(input.flat<T>().data()) +
                i * output_bytes,
            output_bytes, musaMemcpyDeviceToDevice, stream);
        OP_REQUIRES(ctx, err == musaSuccess,
                    errors::Internal("musaMemcpyAsync failed for output ", i,
                                     ": ", musaGetErrorString(err)));
      }
      return;
    }

    const T* input_ptr = input.flat<T>().data();

    // Launch unpack kernel for each output
    for (int i = 0; i < N; ++i) {
      T* output_ptr = outputs[i]->flat<T>().data();
      LaunchUnpackSingleForType(input_ptr, output_ptr, outer_size, N,
                                inner_size, i, stream);
    }
  }

 private:
  void LaunchUnpackSingleForType(const T* input, T* output, int64_t outer_size,
                                 int64_t N, int64_t inner_size,
                                 int64_t unpack_idx, musaStream_t stream);

  int axis_;
  int num_outputs_attr_;
};

// Type-specific unpack launcher implementations
template <>
void MusaUnpackOp<float>::LaunchUnpackSingleForType(
    const float* input, float* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleFloat(input, output, outer_size, N, inner_size, unpack_idx,
                          stream);
}

template <>
void MusaUnpackOp<double>::LaunchUnpackSingleForType(
    const double* input, double* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleDouble(input, output, outer_size, N, inner_size, unpack_idx,
                           stream);
}

template <>
void MusaUnpackOp<int32>::LaunchUnpackSingleForType(
    const int32* input, int32* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleInt32(input, output, outer_size, N, inner_size, unpack_idx,
                          stream);
}

template <>
void MusaUnpackOp<int64>::LaunchUnpackSingleForType(
    const int64* input, int64* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleInt64(input, output, outer_size, N, inner_size, unpack_idx,
                          stream);
}

template <>
void MusaUnpackOp<uint8>::LaunchUnpackSingleForType(
    const uint8* input, uint8* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleUInt8(input, output, outer_size, N, inner_size, unpack_idx,
                          stream);
}

template <>
void MusaUnpackOp<bool>::LaunchUnpackSingleForType(
    const bool* input, bool* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleBool(input, output, outer_size, N, inner_size, unpack_idx,
                         stream);
}

template <>
void MusaUnpackOp<Eigen::half>::LaunchUnpackSingleForType(
    const Eigen::half* input, Eigen::half* output, int64_t outer_size,
    int64_t N, int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleHalf(reinterpret_cast<const void*>(input),
                         reinterpret_cast<void*>(output), outer_size, N,
                         inner_size, unpack_idx, stream);
}

template <>
void MusaUnpackOp<bfloat16>::LaunchUnpackSingleForType(
    const bfloat16* input, bfloat16* output, int64_t outer_size, int64_t N,
    int64_t inner_size, int64_t unpack_idx, musaStream_t stream) {
  LaunchUnpackSingleBFloat16(reinterpret_cast<const void*>(input),
                             reinterpret_cast<void*>(output), outer_size, N,
                             inner_size, unpack_idx, stream);
}

// Kernel Registration
#define REGISTER_MUSA_PACK_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("Pack").Device("MUSA").TypeConstraint<type>("T"), \
      MusaPackOp<type>);

#define REGISTER_MUSA_UNPACK_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Unpack").Device("MUSA").TypeConstraint<type>("T"), \
      MusaUnpackOp<type>);

// Register Pack operators
REGISTER_MUSA_PACK_KERNELS(float)
REGISTER_MUSA_PACK_KERNELS(double)
REGISTER_MUSA_PACK_KERNELS(int32)
REGISTER_MUSA_PACK_KERNELS(int64)
REGISTER_MUSA_PACK_KERNELS(Eigen::half)
REGISTER_MUSA_PACK_KERNELS(bfloat16)
REGISTER_MUSA_PACK_KERNELS(bool)
REGISTER_MUSA_PACK_KERNELS(uint8)

// Register Unpack operators
REGISTER_MUSA_UNPACK_KERNELS(float)
REGISTER_MUSA_UNPACK_KERNELS(double)
REGISTER_MUSA_UNPACK_KERNELS(int32)
REGISTER_MUSA_UNPACK_KERNELS(int64)
REGISTER_MUSA_UNPACK_KERNELS(Eigen::half)
REGISTER_MUSA_UNPACK_KERNELS(bfloat16)
REGISTER_MUSA_UNPACK_KERNELS(bool)
REGISTER_MUSA_UNPACK_KERNELS(uint8)

#undef REGISTER_MUSA_PACK_KERNELS
#undef REGISTER_MUSA_UNPACK_KERNELS

}  // namespace musa
}  // namespace tensorflow
