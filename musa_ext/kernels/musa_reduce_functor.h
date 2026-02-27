#ifndef MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_
#define MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_

#include <functional>
#include <memory>

#include "musa_cast_functor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

struct ReduceFunctor {
  template <typename T>
  static Status Compute(OpKernelContext* ctx, mTensor* output, mTensor* input,
                        ::musa::dnn::Reduce::Mode mode, const int* reduce_dims,
                        int reduce_dim_count, const char* error_prefix) {
    auto& handle = GetHandleByCtx(ctx);

    mReduce op;
    op.SetMode(mode);
    op.SetDim(reduce_dim_count, reduce_dims);

    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    auto alloc_func =
        [tf_allocator](
            size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
      void* ptr = tf_allocator->AllocateRaw(256, size);
      std::function<void(void*)> deleter = [tf_allocator](void* p) {
        if (p) tf_allocator->DeallocateRaw(p);
      };
      return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };
    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    auto status = op.Run(handle, *output, *input, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(error_prefix, static_cast<int>(status));
    }
    return Status::OK();
  }
};

// Given the fact that bf16 does not work well with reduction, we compute the
// reduction in fp32 and cast the result back to bf16.
// This conversion is aligned with tensorflow's convention of promoting bf16 to
// fp32 for ReduceFunctor.
template <>
Status ReduceFunctor::Compute<bfloat16>(OpKernelContext* ctx,
                                        mTensor* output_mt, mTensor* input_mt,
                                        ::musa::dnn::Reduce::Mode mode,
                                        const int* reduce_dims,
                                        int reduce_dim_count,
                                        const char* error_prefix) {
  mTensor input_fp32;
  TF_RETURN_IF_ERROR(CastTensor(ctx, *input_mt, DT_FLOAT, &input_fp32));

  mTensor output_fp32;
  TF_RETURN_IF_ERROR(Compute<float>(ctx, &output_fp32, &input_fp32, mode,
                                    reduce_dims, reduce_dim_count,
                                    error_prefix));

  return CastTensor(ctx, output_fp32, DataTypeToEnum<bfloat16>::value,
                    output_mt);
}

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_