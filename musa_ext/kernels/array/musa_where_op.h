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
void LaunchMusaSelectFlaggedKernel(const T* input, TIndex* selected_indices,
                                   TIndex* num_selected_out, int num_items,
                                   musaStream_t stream);

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
    TF_RETURN_IF_ERROR(
      ctx->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({1}),
                 &count64_wrapper));
    TIndex* count_device = count64_wrapper.flat<TIndex>().data();

    LaunchIsNonZeroCount<T, TIndex>(input_data, count_device,
                            static_cast<int>(input.size()), mstream);

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
    std::size_t temp_storage_bytes = 0;

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                                          TensorShape({1}), &found_true_t));
    TIndex* found_true_device = found_true_t.flat<TIndex>().data();

    // MUSA path: allocate temporary flat buffer for selected indices and
    // perform a simple selection kernel that writes matching indices into the
    // buffer and increments the device-side counter. Then expand indices into
    // the NDIM output using the existing propagate kernel.
    Tensor selected_indices_t;
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                           TensorShape({static_cast<int64_t>(input.size())}),
                           &selected_indices_t));
    TIndex* selected_indices = selected_indices_t.flat<TIndex>().data();

    // Initialize counter to zero on device.
    musaError_t m_err =
        musaMemsetAsync(found_true_device, 0, sizeof(TIndex), stream);
    if (m_err != musaSuccess) {
      return errors::Internal("WhereOp: musaMemsetAsync failed: ",
                              musaGetErrorString(m_err));
    }

    LaunchMusaSelectFlaggedKernel<T, TIndex>(
        input.data(), selected_indices, found_true_device,
        static_cast<int>(input.size()), stream);

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
