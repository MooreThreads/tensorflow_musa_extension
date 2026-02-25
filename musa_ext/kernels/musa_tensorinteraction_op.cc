#include <mublas.h>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaInteractOp : public OpKernel {
 public:
  explicit MusaInteractOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(
        ctx, input.dims() == 3,
        errors::InvalidArgument("Input must be a 3-D tensor [Batch, N, D]"));

    const int64 batch_size = input.dim_size(0);
    const int64 num_features = input.dim_size(1);  // N
    const int64 embed_dim = input.dim_size(2);     // D

    TensorShape output_shape({batch_size, num_features, num_features});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (input.NumElements() == 0) return;

    auto* device = GetDeviceByCtx(ctx);

    mublasHandle_t blas_handle = device->mublas_handle();
    int m = num_features;  // N
    int n = num_features;  // N
    int k = embed_dim;     // D

    float alpha = 1.0f;
    float beta = 0.0f;

    long long strideA = num_features * embed_dim;
    long long strideB = strideA;
    long long strideC = num_features * num_features;

    const T* d_A = input.flat<T>().data();
    const T* d_B = input.flat<T>().data();
    T* d_C = output->flat<T>().data();

    mublasStatus_t status;

    if (std::is_same<T, float>::value) {
      status = mublasSgemmStridedBatched(
          blas_handle,
          MUBLAS_OP_T,                                     // Op(A)
          MUBLAS_OP_N,                                     // Op(B)
          m, n, k, &alpha, (const float*)d_A, k, strideA,  // lda = k
          (const float*)d_B, k, strideB,                   // ldb = k
          &beta, (float*)d_C, m, strideC,                  // ldc = m
          batch_size);
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::Unimplemented("Only float32 is supported in this demo."));
      return;
    }

    OP_REQUIRES(ctx, status == MUBLAS_STATUS_SUCCESS,
                errors::Internal("muBLAS SgemmStridedBatched failed. Status: ",
                                 status));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MusaInteract").Device("MUSA").TypeConstraint<float>("T"),
    MusaInteractOp<float>);

}  // namespace musa
}  // namespace tensorflow

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("MusaInteract")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));

      ::tensorflow::shape_inference::DimensionHandle batch =
          c->Dim(input_shape, 0);
      ::tensorflow::shape_inference::DimensionHandle n = c->Dim(input_shape, 1);

      c->set_output(0, c->MakeShape({batch, n, n}));
      return Status::OK();
    });
}
