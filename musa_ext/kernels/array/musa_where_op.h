#ifndef TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
#define TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_

#include <array>
#include <cstdint>
#include <type_traits>

#include "../math/musa_reduce_functor.h"
#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace musa {

// Count Non Zero within the input tensor
template <typename T, typename TIndex>
void LaunchIsNonZeroCount(const T* input, TIndex* output, int n,
                          musaStream_t stream);

template <typename T, typename TIndex>
void LaunchMusaMarkFlaggedKernel(const T* input, TIndex* d_marks, int num_items,
                                 musaStream_t stream);

template <typename T, typename TIndex>
void LaunchMusaSelectFlaggedKernel(const T* input, TIndex* selected_indices,
                                   const TIndex* d_scanned,
                                   const TIndex* d_marks, int num_items,
                                   int output_size, musaStream_t stream);

template <int NDIM, typename TIndex>
void LaunchPropagateWhereIndicesKernel(const TIndex output_rows,
                                       const TIndex* strides_host,
                                       const TIndex* selected_indices,
                                       TIndex* output, musaStream_t stream);

template <typename T, typename TIndex>
struct NumTrue {
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T>::ConstFlat input,
                        typename TTypes<TIndex>::UnalignedScalar num_true) {
    musaStream_t mstream = GetMusaStreamByCtx(ctx);
    const T* input_data = reinterpret_cast<const T*>(input.data());
    TIndex* num_true_data = num_true.data();

    if (input.size() == 0) {
      *num_true_data = static_cast<TIndex>(0);
      return Status::OK();
    }

    // Use the new LaunchIsNonZeroCount operator which directly counts
    // non-zero values into a 64-bit device scalar, then copy/truncate the
    // result into the requested `TIndex` device scalar.
    Tensor count64_wrapper;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                                          TensorShape({1}), &count64_wrapper));
    TIndex* count_device = count64_wrapper.flat<TIndex>().data();

    LaunchIsNonZeroCount<T, TIndex>(input_data, count_device,
                                    static_cast<int>(input.size()), mstream);

    // Use synchronous memcpy for small data (single scalar)
    // This is acceptable because the data size is tiny (sizeof(TIndex))
    // and we need the result immediately on CPU for subsequent allocations
    auto m_err = musaMemcpy(num_true_data, count_device, sizeof(TIndex),
                            musaMemcpyDeviceToHost);
    if (m_err != musaSuccess) {
      return errors::Internal("WhereOp: musaMemcpy failed: ",
                              musaGetErrorString(m_err));
    }

    return Status::OK();
  }
};

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

// Be advised: The original TF implementation has an extra template parameter
// called `IsConvertibleToBool`, which considered data types that cannot be
// directly converted to bool, namely complex types. For now we only consider
// real number cases.
struct Where {
  template <int NDIM, typename T, typename TIndex>
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T, NDIM>::ConstTensor input,
                        typename TTypes<TIndex>::Matrix output) {
    if (output.dimension(0) == 0) {
      return Status::OK();
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    const int64_t num_items = input.size();

    // Turn the inputted tensor into 0/1 flags (element-wise).
    Tensor marks_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                                          TensorShape({num_items}), &marks_t));
    TIndex* d_marks = marks_t.flat<TIndex>().data();
    LaunchMusaMarkFlaggedKernel<T, TIndex>(input.data(), d_marks,
                                           static_cast<int>(num_items), stream);

    // Compute Prefix Sum using muDNN mCum
    Tensor scanned_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<TIndex>::value, TensorShape({num_items}), &scanned_t));
    TIndex* d_scanned = scanned_t.flat<TIndex>().data();

    auto& handle = GetHandleByCtx(ctx);
    mTensor t_marks = CreateMTensor(marks_t);
    mTensor t_scanned = CreateMTensor(scanned_t);
    mCum cum_op;
    // Set mode: Sum, Dimension: 0
    cum_op.SetMode(::musa::dnn::Cum::Mode::ADD);
    int dim = 0;
    cum_op.SetDim(dim);

    std::list<Tensor> workspace_tensors;
    auto mem_alloc_func =
        [ctx, &workspace_tensors](size_t size) -> ::musa::dnn::MemoryHandler {
      workspace_tensors.emplace_back();
      Tensor& temp = workspace_tensors.back();

      Status s = ctx->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return nullptr;

      void* raw_ptr = static_cast<void*>(temp.flat<uint8_t>().data());
      return ::musa::dnn::MemoryHandler(raw_ptr, [](void* p) {});
    };
    ::musa::dnn::MemoryMaintainer maintainer =
        MakeMusaMemMaintainer(mem_alloc_func);

    // muDNN CumSum handles the global scan
    mStatus status = cum_op.Run(handle, t_scanned, t_marks, maintainer);
    if (status != mStatus::SUCCESS) {
      return errors::Internal("WhereOp: muDNN CumSum failed with status ",
                              (int)status);
    }

    // Extract indices based on prefix sum
    Tensor selected_indices_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<TIndex>::value,
        TensorShape({static_cast<int64_t>(output.dimension(0))}),
        &selected_indices_t));
    TIndex* selected_indices = selected_indices_t.flat<TIndex>().data();

    LaunchMusaSelectFlaggedKernel<T, TIndex>(
        input.data(), selected_indices, d_scanned, d_marks,
        static_cast<int>(num_items), static_cast<int>(output.dimension(0)),
        stream);

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    const TIndex output_rows = output.dimension(0);
    LaunchPropagateWhereIndicesKernel<NDIM, TIndex>(
        output_rows, strides.data(), selected_indices, output.data(), stream);

    return Status::OK();
  }
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
