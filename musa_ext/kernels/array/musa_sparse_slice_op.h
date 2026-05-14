#ifndef TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_SPARSE_SLICE_OP_H_
#define TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_SPARSE_SLICE_OP_H_

#include <mudnn.h>

#include <cstdint>
#include <list>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchSparseSliceScatterKernel(const int64_t* indices, const T* values,
                                    const int64_t* start, const int64_t* marks,
                                    const int64_t* scanned,
                                    int64_t* output_indices, T* output_values,
                                    int64_t nnz, int rank, musaStream_t stream);

void LaunchSparseSliceMarkKernel(const int64_t* indices, const int64_t* start,
                                 const int64_t* size, int64_t* marks,
                                 int64_t nnz, int rank, musaStream_t stream);

inline Status SparseSlicePrefixSum(OpKernelContext* ctx, const Tensor& marks,
                                   Tensor* scanned) {
  if (QueryMusaKernelRuntimeView(ctx).mudnn_handle == nullptr) {
    return MusaMudnnHandleRequiredError();
  }
  auto& handle = GetHandleByCtx(ctx);
  mTensor marks_mt = CreateMTensor(marks);
  mTensor scanned_mt = CreateMTensor(*scanned);

  mCum cum_op;
  cum_op.SetMode(::musa::dnn::Cum::Mode::ADD);
  cum_op.SetDim(0);

  std::list<Tensor> workspace_tensors;
  auto mem_alloc_func =
      [ctx, &workspace_tensors](size_t size) -> ::musa::dnn::MemoryHandler {
    workspace_tensors.emplace_back();
    Tensor& temp = workspace_tensors.back();
    Status s = ctx->allocate_temp(
        DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
    if (!s.ok()) return nullptr;
    void* raw_ptr = static_cast<void*>(temp.flat<uint8>().data());
    return ::musa::dnn::MemoryHandler(raw_ptr, [](void* p) {});
  };
  ::musa::dnn::MemoryMaintainer maintainer(mem_alloc_func);

  mStatus status = cum_op.Run(handle, scanned_mt, marks_mt, maintainer);
  if (status != mStatus::SUCCESS) {
    return errors::Internal("SparseSlice: muDNN CumSum failed with status ",
                            static_cast<int>(status));
  }
  return Status::OK();
}

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_SPARSE_SLICE_OP_H_
