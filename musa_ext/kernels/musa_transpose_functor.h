#include <mudnn.h>

#include "utils_op.h"

namespace tensorflow {
namespace musa {

void DoTranspose(OpKernelContext* ctx, mTensor& in_mt,
                 const std::vector<int64_t>& permutation, mTensor& out_mt) {
  mHandle& h = GetHandleByCtx(ctx);

  ::musa::dnn::Permute pop;

  if (::musa::dnn::Status::SUCCESS !=
      pop.ConfigDimStride(out_mt, in_mt, static_cast<int>(permutation.size()),
                          permutation.data())) {
    ctx->CtxFailure(errors::Internal("muDNN Permute ConfigDimStride failed!"));
    return;
  }

  if (::musa::dnn::Status::SUCCESS != pop.Run(h, out_mt, in_mt)) {
    ctx->CtxFailure(errors::Internal("muDNN Permute Run failed!"));
    return;
  }
}

}  // namespace musa
}  // namespace tensorflow