#include <musa_runtime.h>

#include "../math/musa_reduce_functor.h"


namespace tensorflow {
namespace musa {

template <typename DstT>
void LaunchBoolCast(const bool* src, DstT* dst, int n, musaStream_t stream);

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,
    int64* __restrict__ output) {
  // TODO(ebrevdo): Use a multi-dimensional loop, increasing the
  // dimensions of individual indices manually, instead of relying on
  // a scalar loop variable and using integer division.
  GPU_1D_KERNEL_LOOP(i, output_rows) {
    TIndex index_value = ldg(output + NDIM * i);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];
      index_value %= strides[c];
    }
  }
}

template <typename T, typename TIndex>
struct CubDeviceReduceCount {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
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
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
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
    TIndex* num_true_data = num_true.data();

    if (input.size() == 0) {
      *num_true_data = static_cast<TIndex>(0);
      return OkStatus();
    }

    if constexpr (std::is_same<T, bool>::value) {
      Tensor input_i32_t;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_INT32, TensorShape({static_cast<int64_t>(input.size())}),
          &input_i32_t));

      LaunchBoolCast<int32_t>(input.data(), input_i32_t.flat<int32>().data(),
                              static_cast<int>(input.size()), mstream);
      auto cast_launch_status = musaGetLastError();
      if (cast_launch_status != musaSuccess) {
        return errors::Internal("WhereOp NumTrue bool cast launch failed: ",
                                musaGetErrorString(cast_launch_status));
      }

      Tensor output_i32_t;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, TensorShape({}),
                                            &output_i32_t));

      mTensor input_mt = CreateMTensor(input_i32_t);
      mTensor output_mt = CreateMTensor(output_i32_t);
      int reduce_dims[] = {0};
      TF_RETURN_IF_ERROR(ReduceFunctor::Compute<int32_t>(
          ctx, &output_mt, &input_mt, ::musa::dnn::Reduce::Mode::ADD,
          reduce_dims, 1,
          "WhereOp NumTrue ReduceFunctor(bool->int32 sum) failed. Status: "));

      int32 h_num_true = 0;
      auto memcpy_status = musaMemcpyAsync(&h_num_true,
                                           output_i32_t.scalar<int32>().data(),
                                           sizeof(int32), musaMemcpyDeviceToHost,
                                           mstream);
      if (memcpy_status != musaSuccess) {
        return errors::Internal(
            "WhereOp NumTrue memcpy(bool reduce result) failed: ",
            musaGetErrorString(memcpy_status));
      }
      auto sync_status = musaStreamSynchronize(mstream);
      if (sync_status != musaSuccess) {
        return errors::Internal("WhereOp NumTrue stream sync failed: ",
                                musaGetErrorString(sync_status));
      }
      *num_true_data = static_cast<TIndex>(h_num_true);
      return OkStatus();
    }

    std::size_t temp_storage_bytes = 0;
    Tensor num_true_device_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::v(),
                                          TensorShape({}), &num_true_device_t));
    TIndex* num_true_device_data = num_true_device_t.scalar<TIndex>().data();

    CubDeviceReduceCount<T, TIndex> counter;
    auto first_success =
        counter(nullptr, temp_storage_bytes, input.data(), num_true_device_data,
                static_cast<int>(input.size()), mstream);
    if (first_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp NumTrue: Could not launch gpuprim::DeviceReduce::Sum to "
          "calculate temp_storage_bytes, status: ",
          GpuGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success =
        counter(temp_storage.flat<int8>().data(), temp_storage_bytes,
                input.data(), num_true_device_data,
                static_cast<int>(input.size()), mstream);
    if (second_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp NumTrue: Could not launch gpuprim::DeviceReduce::Sum to "
          "count nonzero, status: ",
          GpuGetErrorString(second_success));
    }

    auto memcpy_status = musaMemcpyAsync(num_true_data, num_true_device_data,
                                         sizeof(TIndex), musaMemcpyDeviceToHost,
                                         mstream);
    if (memcpy_status != musaSuccess) {
      return errors::Internal("WhereOp NumTrue memcpy failed: ",
                              musaGetErrorString(memcpy_status));
    }
    auto sync_status = musaStreamSynchronize(mstream);
    if (sync_status != musaSuccess) {
      return errors::Internal("WhereOp NumTrue stream sync failed: ",
                              musaGetErrorString(sync_status));
    }
    return OkStatus();
  }
};

template <int NDIM>
class WhereOutputIterator {
 public:
  // Required iterator traits
  typedef WhereOutputIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void pointer;
  typedef int64& reference;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  WhereOutputIterator(int64* ptr, const Eigen::DenseIndex max_row)
      : ptr_(ptr), max_row_(max_row) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64& operator[](int n) const {
    // If the selection mechanism finds too many true values (because
    // the input tensor changed between allocation of output and now),
    // we may accidentally try to write past the allowable memory.  If
    // valid is false, then we don't do this.  Instead, we'll read off
    // the number of items found in Flagged()'s d_num_selected_out at
    // the end and confirm that it matches the number of rows of output.
    const bool valid = FastBoundsCheck(n, max_row_);
    return *(ptr_ + (valid ? (NDIM * n) : 0));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE reference operator*() const {
    // Dereference the current pointer
    return *ptr_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE self_type
  operator+(std::ptrdiff_t n) const {
    return self_type(ptr_ + NDIM * n, max_row_);
  }

 private:
  int64* ptr_;
  const Eigen::DenseIndex max_row_;
};

template <typename T, typename TIndex, typename OutputIterator,
          bool IsConvertibleToBool>
struct CubDeviceSelectFlaggedCounter;

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     false /*IsConvertibleToBool*/> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, OutputIterator d_out,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    IsNonzero<T> is_nonzero;
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_flags, is_nonzero);
    return gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/,
        is_nonzero_iter /*d_flags*/, d_out, d_num_selected_out, num_items,
        stream);
  }
};

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     true /*IsConvertibleToBool*/> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, OutputIterator d_out,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    return gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/, d_flags,
        d_out, d_num_selected_out, num_items, stream);
  }
};

struct Where {
  template <int NDIM, typename T, typename TIndex>
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T, NDIM>::ConstTensor input,
                        typename TTypes<int64_t>::Matrix output,
                        TIndex* found_true_host) {
    if (output.dimension(0) == 0) {
      // Nothing to do.
      return OkStatus();
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    std::size_t temp_storage_bytes = 0;

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::v(),
                                          TensorShape({}), &found_true_t));
    TIndex* found_true_device = found_true_t.scalar<TIndex>().data();

    WhereOutputIterator<NDIM> output_iterator(
        output.data(),
        /* max_row */ output.dimension(0));

    typedef std::decay<T> DT;
    CubDeviceSelectFlaggedCounter<
        T, TIndex, decltype(output_iterator) /*OutputIterator*/,
        std::is_convertible<DT, bool>::value /*IsConvertibleToBool*/>
        counter;
    auto first_success = counter(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_flags*/ input.data(),
                                 /*d_out*/ output_iterator,
                                 /*d_num_selected_out*/ found_true_device,
                                 /*num_items*/ input.size(),
                                 /*stream*/ stream);
    if (first_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to "
          "calculate "
          "temp_storage_bytes, status: ",
          GpuGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = counter(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_flags*/ input.data(),
        /*d_out*/ output_iterator,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ input.size(),
        /*stream*/ stream);

    if (second_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to copy "
          "indices out, status: ",
          GpuGetErrorString(second_success));
    }

    // TODO(ebrevdo): Find a way to synchronously copy back data from
    // found_true_device to *found_true_host.

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    const TIndex output_rows = output.dimension(0);
    GpuLaunchConfig config = GetGpuLaunchConfig(output_rows, d);
    TF_CHECK_OK(GpuLaunchKernel(PropagateWhereIndicesKernel<NDIM, TIndex>,
                                config.block_count, config.thread_per_block, 0,
                                stream, output_rows, strides, output.data()));

    return OkStatus();
  }
};

}  // namespace musa
}  // namespace tensorflow