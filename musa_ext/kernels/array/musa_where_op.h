#include "musa_where_kernel.mu"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace musa {

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
    const T* input_data = input.data();
    TIndex* num_true_data = num_true.data();

    if (input.size() == 0) {
      *num_true_data = static_cast<TIndex>(0);
      return Status::OK();
    }

    std::size_t temp_storage_bytes = 0;
    auto reducer = CubDeviceReduceCount<T, TIndex>();
    auto first_success = reducer(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_in*/ input_data,
                                 /*d_out*/ num_true_data,
                                 /*num_items*/ input.size(),
                                 /*stream*/ mstream);

    if (first_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to calculate "
          "temp_storage_bytes.");
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = reducer(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in*/ input_data,
        /*d_out*/ num_true_data,
        /*num_items*/ input.size(),
        /*stream*/ mstream);

    if (second_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to count "
          "number of true / nonzero indices.  temp_storage_bytes: ",
          temp_storage_bytes, ".");
    }

    return Status::OK();
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
      return Status::OK();
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

    auto propagate_status = LaunchPropagateWhereIndicesKernel(
        output_rows, strides, output.data(), stream);
    if (!propagate_status.ok()) {
      return errors::Internal(
          "WhereOp: Failed to launch PropagateWhereIndicesKernel, status: ",
          propagate_status.ToString());
    }
    return Status::OK();
  }

  Status LaunchPropagateWhereIndicesKernel(
      int64 output_rows, const Eigen::array<int64, 4>& strides, int64* output,
      musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (output_rows + block_size - 1) / block_size;
    PropagateWhereIndicesKernel<4, int64>
        <<<grid_size, block_size, 0, stream>>>(output_rows, strides, output);
    auto launch_status = musaGetLastError();
    if (launch_status != musaSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch PropagateWhereIndicesKernel, status: ",
          musaGetErrorString(launch_status));
    }
    return Status::OK();
  };

}  // namespace musa
}  // namespace tensorflow