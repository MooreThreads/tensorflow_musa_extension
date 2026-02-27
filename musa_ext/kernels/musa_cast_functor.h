#include <functional>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

static Status CastTensor(OpKernelContext* ctx, const mTensor& input_mt,
                         DataType dst_dtype, mTensor* output_mt) {
  ::musa::dnn::Unary op;
  auto status = op.SetMode(::musa::dnn::Unary::Mode::CAST);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("Einsum CastTensor SetMode failed. Status: ",
                            static_cast<int>(status));
  }
  status = op.Run(GetHandleByCtx(ctx), *output_mt, input_mt);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("Einsum CastTensor Run failed. Status: ",
                            static_cast<int>(status));
  }
  return Status::OK();
}

}  // namespace musa
}  // namespace tensorflow