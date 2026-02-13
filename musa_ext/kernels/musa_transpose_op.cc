#include <mudnn.h>

#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaTransposeOp : public MusaOpKernel {
 public:
  explicit MusaTransposeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& perm_tensor = ctx->input(1);
    const int dims = input.dims();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm_tensor.shape()),
                errors::InvalidArgument("perm must be rank 1"));
    OP_REQUIRES(ctx, dims == perm_tensor.NumElements(),
                errors::InvalidArgument("transpose expects a vector of size ",
                                        input.dims(),
                                        ". But input(1) is a vector of size ",
                                        perm_tensor.NumElements()));

    std::vector<int64_t> permutation_64;
    permutation_64.reserve(dims);

    std::vector<bool> bits(dims, false);
    bool is_identity = true;
    TensorShape output_shape;

    auto process_perm = [&](int64_t d, int i) {
      OP_REQUIRES(
          ctx, d >= 0 && d < dims,
          errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      OP_REQUIRES(
          ctx, !bits[d],
          errors::InvalidArgument(d, " is duplicated in the permutation."));

      bits[d] = true;
      permutation_64.push_back(d);
      output_shape.AddDim(input.dim_size(d));

      if (d != i) {
        is_identity = false;
      }
    };

    if (perm_tensor.dtype() == DT_INT32) {
      auto Vperm = perm_tensor.vec<int32>();
      for (int i = 0; i < dims; ++i) process_perm(Vperm(i), i);
    } else {
      auto Vperm = perm_tensor.vec<int64_t>();
      for (int i = 0; i < dims; ++i) process_perm(Vperm(i), i);
    }

    if (!ctx->status().ok()) return;

    if (dims <= 1 || is_identity) {
      ctx->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    mHandle& h = GetHandleByCtx(ctx);

    mTensor in_mt = CreateMTensor(input, format_);
    mTensor out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Permute pop;

    if (::musa::dnn::Status::SUCCESS !=
        pop.ConfigDimStride(out_mt, in_mt,
                            static_cast<int>(permutation_64.size()),
                            permutation_64.data())) {
      ctx->CtxFailure(
          errors::Internal("muDNN Permute ConfigDimStride failed!"));
      return;
    }

    if (::musa::dnn::Status::SUCCESS != pop.Run(h, out_mt, in_mt)) {
      ctx->CtxFailure(errors::Internal("muDNN Permute Run failed!"));
      return;
    }
  }
};

#define REGISTER_MUSA_TRANSPOSE(TYPE)                    \
  REGISTER_KERNEL_BUILDER(Name("Transpose")              \
                              .Device("MUSA")            \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("perm"),       \
                          MusaTransposeOp<TYPE>);

REGISTER_MUSA_TRANSPOSE(float);
REGISTER_MUSA_TRANSPOSE(double);
REGISTER_MUSA_TRANSPOSE(Eigen::half);
REGISTER_MUSA_TRANSPOSE(bfloat16);
REGISTER_MUSA_TRANSPOSE(int32);
REGISTER_MUSA_TRANSPOSE(int64);
REGISTER_MUSA_TRANSPOSE(bool);

#undef REGISTER_MUSA_TRANSPOSE

}  // namespace musa
}  // namespace tensorflow
