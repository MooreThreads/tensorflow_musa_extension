#ifndef TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
#define TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_

#include <array>
#include <cstdint>
#include <type_traits>

#include "../utils_op.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace musa {

template <int NDIM, typename TIndex>
musaError_t LaunchPropagateWhereIndicesKernel(
    const TIndex output_rows, const TIndex* strides_host,
    const TIndex* selected_indices, int64_t* output, musaStream_t stream);

template <typename T, typename TIndex>
struct CubDeviceReduceCount {
  musaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0) {
    IsNonzero<T> is_nonzero;
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_in, is_nonzero);
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      is_nonzero_iter, d_out, num_items,
                                      stream);
  }
};

template <typename TIndex>
struct CubDeviceReduceCount<bool, TIndex> {
  musaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const bool* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0) {
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                      d_out, num_items, stream);
  }
};

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

    std::size_t temp_storage_bytes = 0;
    auto reducer = CubDeviceReduceCount<T, TIndex>();
    void* temp_storage_ptr = nullptr;
    auto first_success = reducer(/*temp_storage*/ temp_storage_ptr,
                   temp_storage_bytes,
                                 /*d_in*/ input_data,
                                 /*d_out*/ num_true_data,
                                 /*num_items*/ static_cast<int>(input.size()),
                   /*stream*/ static_cast<gpuStream_t>(mstream));

    if (first_success != musaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to calculate "
          "temp_storage_bytes.");
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = reducer(
      /*temp_storage*/ reinterpret_cast<void*>(temp_storage.flat<int8>().data()),
      temp_storage_bytes,
        /*d_in*/ input_data,
        /*d_out*/ num_true_data,
        /*num_items*/ static_cast<int>(input.size()),
      /*stream*/ static_cast<gpuStream_t>(mstream));

    if (second_success != musaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to count "
          "number of true / nonzero indices. temp_storage_bytes: ",
          temp_storage_bytes, ".");
    }

    return Status::OK();
  }
};

template <typename T, typename TIndex, bool IsBoolFlags>
struct CubDeviceSelectIndices;

template <typename T, typename TIndex>
struct CubDeviceSelectIndices<T, TIndex, false> {
  musaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, TIndex* d_selected_indices,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    IsNonzero<T> is_nonzero;
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_flags, is_nonzero);
    auto status = gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/,
        is_nonzero_iter /*d_flags*/, d_selected_indices, d_num_selected_out,
        num_items, stream);
    if (status !=)
  }
};

template <typename T, typename TIndex>
struct CubDeviceSelectIndices<T, TIndex, true> {
  musaError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, TIndex* d_selected_indices,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    return gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/,
        d_flags /*d_flags*/, d_selected_indices, d_num_selected_out, num_items,
        stream);
  }
};

struct Where {
  template <int NDIM, typename T, typename TIndex>
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T, NDIM>::ConstTensor input,
                        typename TTypes<int64_t>::Matrix output) {
    if (output.dimension(0) == 0) {
      return Status::OK();
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    Tensor selected_indices_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<TIndex>::value, TensorShape({output.dimension(0)}),
        &selected_indices_t));
    TIndex* selected_indices = selected_indices_t.flat<TIndex>().data();

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                                          TensorShape({}), &found_true_t));
    TIndex* found_true_device = found_true_t.scalar<TIndex>().data();

    std::size_t temp_storage_bytes = 0;
    using DT = typename std::decay<T>::type;
    CubDeviceSelectIndices<T, TIndex, std::is_same<DT, bool>::value> selector;

    const T* flags_ptr = reinterpret_cast<const T*>(input.data());

    void* temp_storage_ptr = nullptr;
    auto first_success = selector(
      /*temp_storage*/ temp_storage_ptr, temp_storage_bytes,
      /*d_flags*/ flags_ptr,
        /*d_selected_indices*/ selected_indices,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ static_cast<int>(input.size()),
      /*stream*/ static_cast<gpuStream_t>(stream));
    if (first_success != musaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to "
          "calculate temp_storage_bytes, status: ",
          musaGetErrorString(static_cast<musaError_t>(first_success)));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = selector(
      /*temp_storage*/ reinterpret_cast<void*>(temp_storage.flat<int8>().data()),
      temp_storage_bytes,
      /*d_flags*/ flags_ptr,
        /*d_selected_indices*/ selected_indices,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ static_cast<int>(input.size()),
      /*stream*/ static_cast<gpuStream_t>(stream));
    if (second_success != musaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to copy "
          "indices out, status: ",
          musaGetErrorString(static_cast<musaError_t>(second_success)));
    }

    std::array<TIndex, NDIM> strides;
    auto dims = input.dimensions();
    strides[NDIM - 1] = static_cast<TIndex>(1);
    for (int i = NDIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * static_cast<TIndex>(dims[i + 1]);
    }

    const TIndex output_rows = static_cast<TIndex>(output.dimension(0));
    auto launch_status = LaunchPropagateWhereIndicesKernel<NDIM, TIndex>(
        output_rows, strides.data(), selected_indices, output.data(), stream);
    if (launch_status != musaSuccess) {
      return errors::Internal(
          "WhereOp: Failed to launch PropagateWhereIndicesKernel, status: ",
          musaGetErrorString(launch_status));
    }

    return Status::OK();
  }
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
