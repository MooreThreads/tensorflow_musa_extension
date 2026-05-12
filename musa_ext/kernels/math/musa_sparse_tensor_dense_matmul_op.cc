#include <cstdint>
#include <limits>

#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace musa {

namespace {

inline bool FitsInInt64Mul(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return true;
  return a <= std::numeric_limits<int64_t>::max() / b;
}

}  // namespace

template <typename T>
musaError_t LaunchSparseTensorDenseMatMul(
    const int64* a_indices, const T* a_values, const T* b, T* output,
    int64_t nnz, int64_t a_rows, int64_t a_cols, int64_t b_rows,
    int64_t b_cols, int64_t out_rows, int64_t out_cols, bool adjoint_a,
    bool adjoint_b, musaStream_t stream);

template <typename T>
class MusaSparseTensorDenseMatMulOp : public MusaOpKernel {
 public:
  explicit MusaSparseTensorDenseMatMulOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_b", &adjoint_b_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a_indices = ctx->input(0);
    const Tensor& a_values = ctx->input(1);
    const Tensor& a_shape = ctx->input(2);
    const Tensor& b = ctx->input(3);

    OP_REQUIRES(ctx, a_indices.dims() == 2 && a_indices.dim_size(1) == 2,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: a_indices must be a matrix with "
                    "shape [nnz, 2], got ",
                    a_indices.shape().DebugString()));
    OP_REQUIRES(ctx, a_values.dims() == 1,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: a_values must be a vector, got ",
                    a_values.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_shape.shape()) &&
                             a_shape.NumElements() == 2,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: a_shape must be a vector with 2 "
                    "elements, got ",
                    a_shape.shape().DebugString()));
    OP_REQUIRES(ctx, b.dims() == 2,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: b must be rank 2, got ",
                    b.shape().DebugString()));

    const int64_t nnz = a_indices.dim_size(0);
    OP_REQUIRES(ctx, a_values.NumElements() == nnz,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: a_values length must match "
                    "a_indices rows. a_values length=",
                    a_values.NumElements(), ", nnz=", nnz));

    const auto a_shape_vec = a_shape.vec<int64>();
    const int64_t a_rows = a_shape_vec(0);
    const int64_t a_cols = a_shape_vec(1);
    OP_REQUIRES(ctx, a_rows >= 0 && a_cols >= 0,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: a_shape must be non-negative, "
                    "got [",
                    a_rows, ", ", a_cols, "]"));

    const int64_t b_rows = b.dim_size(0);
    const int64_t b_cols = b.dim_size(1);
    const int64_t out_rows = adjoint_a_ ? a_cols : a_rows;
    const int64_t contract_dim = adjoint_a_ ? a_rows : a_cols;
    const int64_t b_contract_dim = adjoint_b_ ? b_cols : b_rows;
    const int64_t out_cols = adjoint_b_ ? b_rows : b_cols;

    OP_REQUIRES(ctx, contract_dim == b_contract_dim,
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: matrix size-incompatible. "
                    "A contraction dim=",
                    contract_dim, ", B contraction dim=", b_contract_dim,
                    ", a_shape=[", a_rows, ", ", a_cols, "], b_shape=",
                    b.shape().DebugString(), ", adjoint_a=", adjoint_a_,
                    ", adjoint_b=", adjoint_b_));
    OP_REQUIRES(ctx, FitsInInt64Mul(nnz, out_cols),
                errors::InvalidArgument(
                    "SparseTensorDenseMatMul: work size overflow. nnz=", nnz,
                    ", out_cols=", out_cols));

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(out_rows));
    OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(out_cols));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    auto stream = GetMusaStreamByCtx(ctx);
    musaError_t status =
        musaMemsetAsync(output->data(), 0, output->TotalBytes(), stream);
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal(
                    "SparseTensorDenseMatMul: failed to zero output: ",
                    musaGetErrorString(status)));

    if (nnz == 0) {
      return;
    }

    status = LaunchSparseTensorDenseMatMul<T>(
        a_indices.flat<int64>().data(), a_values.flat<T>().data(),
        b.flat<T>().data(), output->flat<T>().data(), nnz, a_rows, a_cols,
        b_rows, b_cols, out_rows, out_cols, adjoint_a_, adjoint_b_, stream);
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal(
                    "SparseTensorDenseMatMul MUSA kernel launch failed: ",
                    musaGetErrorString(status)));
  }

 private:
  bool adjoint_a_ = false;
  bool adjoint_b_ = false;
};

#define REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL(TYPE)           \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseMatMul")        \
                              .Device(DEVICE_MTGPU)              \
                              .HostMemory("a_shape")             \
                              .TypeConstraint<TYPE>("T"),        \
                          MusaSparseTensorDenseMatMulOp<TYPE>)

REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL(float);
REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL(double);
REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL(int32);
REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL(bfloat16);

#undef REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL

}  // namespace musa
}  // namespace tensorflow
