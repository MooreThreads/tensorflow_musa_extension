#include "musa_sparse_slice_op.h"

#include <limits>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

namespace {

Status CopyHostInt64ToDevice(OpKernelContext* ctx, const Tensor& src,
                             Tensor* dst, musaStream_t stream) {
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, src.shape(), dst));
  auto err = musaMemcpyAsync(
      dst->flat<int64>().data(), src.flat<int64>().data(),
      src.NumElements() * sizeof(int64), musaMemcpyHostToDevice, stream);
  if (err != musaSuccess) {
    return errors::Internal(
        "SparseSlice: musaMemcpyAsync host to device failed: ",
        musaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace

template <typename T>
class MusaSparseSliceOp : public MusaOpKernel {
 public:
  explicit MusaSparseSliceOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const Tensor& values = ctx->input(1);
    const Tensor& shape = ctx->input(2);
    const Tensor& start = ctx->input(3);
    const Tensor& size = ctx->input(4);

    OP_REQUIRES(ctx, indices.dtype() == DT_INT64,
                errors::InvalidArgument("SparseSlice indices must be int64"));
    OP_REQUIRES(ctx, shape.dtype() == DT_INT64,
                errors::InvalidArgument("SparseSlice shape must be int64"));
    OP_REQUIRES(ctx, start.dtype() == DT_INT64,
                errors::InvalidArgument("SparseSlice start must be int64"));
    OP_REQUIRES(ctx, size.dtype() == DT_INT64,
                errors::InvalidArgument("SparseSlice size must be int64"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices.shape()),
                errors::InvalidArgument("indices must be a matrix, got ",
                                        indices.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values.shape()),
                errors::InvalidArgument("values must be a vector, got ",
                                        values.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(shape.shape()),
                errors::InvalidArgument("shape must be a vector, got ",
                                        shape.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(start.shape()),
                errors::InvalidArgument("start must be a vector, got ",
                                        start.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(size.shape()),
                errors::InvalidArgument("size must be a vector, got ",
                                        size.shape().DebugString()));

    const int64_t nnz = indices.dim_size(0);
    const int64_t rank64 = indices.dim_size(1);
    OP_REQUIRES(ctx,
                rank64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                errors::InvalidArgument("rank is too large: ", rank64));
    const int rank = static_cast<int>(rank64);

    OP_REQUIRES(
        ctx, values.dim_size(0) == nnz,
        errors::InvalidArgument("values length must match indices rows"));
    OP_REQUIRES(
        ctx, shape.dim_size(0) == rank64,
        errors::InvalidArgument("shape length must match indices rank"));
    OP_REQUIRES(
        ctx, start.dim_size(0) == rank64,
        errors::InvalidArgument("start length must match indices rank"));
    OP_REQUIRES(ctx, size.dim_size(0) == rank64,
                errors::InvalidArgument("size length must match indices rank"));

    MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx);
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    Tensor* output_shape = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, TensorShape({rank64}), &output_shape));
    auto output_shape_flat = output_shape->flat<int64>();
    auto size_flat = size.flat<int64>();
    for (int64_t i = 0; i < rank64; ++i) {
      output_shape_flat(i) = size_flat(i);
    }

    if (nnz == 0) {
      Tensor* output_indices = nullptr;
      Tensor* output_values = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({0, rank64}),
                                               &output_indices));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(1, TensorShape({0}), &output_values));
      return;
    }

    Tensor start_device;
    Tensor size_device;
    OP_REQUIRES_OK(ctx,
                   CopyHostInt64ToDevice(ctx, start, &start_device, stream));
    OP_REQUIRES_OK(ctx, CopyHostInt64ToDevice(ctx, size, &size_device, stream));

    Tensor marks;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT64, TensorShape({nnz}), &marks));

    LaunchSparseSliceMarkKernel(indices.flat<int64>().data(),
                                start_device.flat<int64>().data(),
                                size_device.flat<int64>().data(),
                                marks.flat<int64>().data(), nnz, rank, stream);
    auto launch_err = musaGetLastError();
    OP_REQUIRES(ctx, launch_err == musaSuccess,
                errors::Internal("SparseSlice mark kernel launch failed: ",
                                 musaGetErrorString(launch_err)));

    Tensor scanned;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT64, TensorShape({nnz}), &scanned));
    OP_REQUIRES_OK(ctx, SparseSlicePrefixSum(ctx, marks, &scanned));

    int64_t output_nnz = 0;
    auto err =
        musaMemcpyAsync(&output_nnz, scanned.flat<int64>().data() + nnz - 1,
                        sizeof(int64), musaMemcpyDeviceToHost, stream);
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("SparseSlice: copying output count failed: ",
                                 musaGetErrorString(err)));
    err = musaStreamSynchronize(stream);
    OP_REQUIRES(
        ctx, err == musaSuccess,
        errors::Internal("SparseSlice: synchronizing output count failed: ",
                         musaGetErrorString(err)));

    Tensor* output_indices = nullptr;
    Tensor* output_values = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({output_nnz, rank64}),
                                        &output_indices));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({output_nnz}),
                                             &output_values));

    if (output_nnz == 0) return;

    LaunchSparseSliceScatterKernel<T>(
        indices.flat<int64>().data(), values.flat<T>().data(),
        start_device.flat<int64>().data(), marks.flat<int64>().data(),
        scanned.flat<int64>().data(), output_indices->flat<int64>().data(),
        output_values->flat<T>().data(), nnz, rank, stream);
    launch_err = musaGetLastError();
    OP_REQUIRES(ctx, launch_err == musaSuccess,
                errors::Internal("SparseSlice scatter kernel launch failed: ",
                                 musaGetErrorString(launch_err)));
  }
};

#define REGISTER_MUSA_SPARSE_SLICE(type)                   \
  REGISTER_KERNEL_BUILDER(Name("SparseSlice")              \
                              .Device(DEVICE_MTGPU)        \
                              .TypeConstraint<type>("T")   \
                              .HostMemory("start")         \
                              .HostMemory("size")          \
                              .HostMemory("output_shape"), \
                          MusaSparseSliceOp<type>);

REGISTER_MUSA_SPARSE_SLICE(float);
REGISTER_MUSA_SPARSE_SLICE(double);
REGISTER_MUSA_SPARSE_SLICE(int32);
REGISTER_MUSA_SPARSE_SLICE(int64);
REGISTER_MUSA_SPARSE_SLICE(Eigen::half);
REGISTER_MUSA_SPARSE_SLICE(bfloat16);

#undef REGISTER_MUSA_SPARSE_SLICE

}  // namespace musa
}  // namespace tensorflow
