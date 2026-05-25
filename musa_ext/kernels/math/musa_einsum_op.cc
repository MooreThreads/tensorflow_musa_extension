#include <cstdlib>
#include <memory>
#include <vector>

#include "../array/musa_fill_functor.h"
#include "../array/musa_transpose_functor.h"
#include "../utils_op.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "device/musa_device.h"
#include "musa_reduce_functor.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/musa_einsum_op_util.h"

namespace tensorflow {
namespace musa {

extern "C" void LaunchFloatToBFloat16Copy(const float* src, void* dst,
                                           int64_t n, musaStream_t stream);

using ShapeVec = gtl::InlinedVector<int64_t, 8>;
using Labels = gtl::InlinedVector<int, 8>;
using OperandLabels = gtl::InlinedVector<Labels, 2>;
using LabelCounts = gtl::InlinedVector<int, 8>;
using OperandLabelCounts = gtl::InlinedVector<LabelCounts, 2>;
using LabelToDimSizes = gtl::InlinedVector<int64_t, 8>;

struct EinsumHelper {
  // Insert new (unnamed) broadcasting labels at the location of ellipsis.
  static void InsertBroadcastLabels(int num_bcast_dims, int num_named_labels,
                                    int ellipsis_axis, Labels* labels,
                                    LabelCounts* label_counts) {
    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis,
              labels->begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  // Record and validate the label to dimension mapping. Must be a named
  // (non-broadcasting) label as broadcasting labels don't have a fixed
  // dimension.
  static Status RecordLabelToDimension(const int label, const int axis,
                                       const Tensor& input,
                                       LabelToDimSizes* label_to_dim_sizes) {
    const int64_t input_dim = input.dim_size(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes->at(label) != 0 &&
        label_to_dim_sizes->at(label) != input_dim) {
      return errors::InvalidArgument(
          "Expected dimension ", label_to_dim_sizes->at(label), " at axis ",
          axis, " of the input shaped ", input.shape().DebugString(),
          " but got dimension ", input_dim);
    }
    (*label_to_dim_sizes)[label] = input_dim;
    return OkStatus();
  }

  // Validate input dimensions and populate unnamed labels and their label
  // counts.
  static Status ProcessDimensions(
      const OpInputList& inputs,
      const gtl::InlinedVector<bool, 2>& input_has_ellipsis,
      const bool output_has_ellipsis, OperandLabels* input_labels,
      Labels* output_labels, std::vector<EinsumDimensionType>* label_types,
      OperandLabelCounts* input_label_counts, LabelCounts* output_label_counts,
      LabelToDimSizes* label_to_dim_sizes) {
    if (inputs.size() != input_labels->size()) {
      return errors::InvalidArgument("Expected ", input_labels->size(),
                                     " inputs but got: ", inputs.size());
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum
    // rank among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types->size();
    label_to_dim_sizes->resize(num_named_labels);
    for (int i = 0; i < num_inputs; ++i) {
      Labels* labels = &(*input_labels)[i];

      if (!input_has_ellipsis[i]) {
        if (inputs[i].dims() != labels->size()) {
          return errors::InvalidArgument("Expected input ", i, " to have rank ",
                                         labels->size(),
                                         " but got: ", inputs[i].dims());
        }
        for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
          const int label = (*labels)[label_idx];
          TF_RETURN_IF_ERROR(RecordLabelToDimension(label, label_idx, inputs[i],
                                                    label_to_dim_sizes));
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].dims() + 1 < labels->size()) {
        return errors::InvalidArgument(
            "Expected input ", i, " to have rank at least ", labels->size() - 1,
            " but got: ", inputs[i].dims());
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].dims() - labels->size() + 1;
      for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
        const int label = (*labels)[label_idx];
        if (label == kEllipsisLabel) {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis =
            label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        TF_RETURN_IF_ERROR(
            RecordLabelToDimension(label, axis, inputs[i], label_to_dim_sizes));
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1) {
        InsertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis,
                              labels, &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }
    if (!absl::c_linear_search(input_has_ellipsis, true) &&
        !output_has_ellipsis) {
      return OkStatus();
    }
    // Insert broadcasting dimensions in the output labels.
    auto it =
        std::find(output_labels->begin(), output_labels->end(), kEllipsisLabel);
    if (it != output_labels->end()) {
      const int ellipsis_axis = it - output_labels->begin();
      InsertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis,
                            output_labels, output_label_counts);
    } else if (max_bcast_dims > 0) {
      return errors::InvalidArgument(
          "Output contains ", max_bcast_dims,
          " broadcasting dimension(s) but no ellipsis "
          "(...) was found in the output subscripts.");
    }
    // Populate EinsumDimensionType for the new broadcasting labels.
    label_types->resize(num_named_labels + max_bcast_dims,
                        EinsumDimensionType::kBroadcasting);
    return OkStatus();
  }

  // Permutes the labels according to the given permutation.
  static void PermuteLabels(const std::vector<int64_t>& permutation,
                            Labels* labels) {
    Labels permuted_labels(labels->size());
    for (int i = 0; i < labels->size(); ++i) {
      permuted_labels[i] = (*labels)[permutation[i]];
    }
    labels->swap(permuted_labels);
  }

  // Returns a reshaped input Tensor. The underlying buffer is not copied.
  static Status CopyFrom(const Tensor& input, const TensorShape& shape,
                         Tensor* output) {
    if (output->CopyFrom(input, shape)) return OkStatus();
    return errors::Internal(
        "Encountered error while reshaping a Tensor of shape ",
        input.shape().DebugString(), " to shape ", shape.DebugString());
  }

  // Returns whether transposing would be a no-op; whether input has rank < 2 or
  // the permutation is the identity permutation.
  static bool ShouldTranspose(const TensorShape& input_shape,
                              const std::vector<int64_t>& permutation) {
    if (input_shape.dims() < 2) return false;
    for (int i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return true;
    }
    return false;
  }

  static Status CastTensor(OpKernelContext* ctx, const Tensor& input,
                           DataType dst_dtype, Tensor* output) {
    if (input.dtype() == dst_dtype) {
      return CopyFrom(input, input.shape(), output);
    }

    TF_RETURN_IF_ERROR(ctx->allocate_temp(dst_dtype, input.shape(), output));
    if (input.NumElements() == 0) return OkStatus();

    auto input_mt = CreateMTensor(input);
    auto output_mt = CreateMTensor(*output);

    return CastFunctor(ctx, input_mt, &output_mt);
  }

  // Transpose the input given a permutation. Returns a reference to the input
  // if transposing is not necessary.
  template <typename T>
  static Status TransposeOperand(OpKernelContext* ctx, const Tensor& input,
                                 const std::vector<int64_t>& permutation,
                                 Tensor* output) {
    if (!ShouldTranspose(input.shape(), permutation)) {
      return CopyFrom(input, input.shape(), output);
    }
    TensorShape transposed_shape;
    for (int i = 0; i < input.dims(); ++i) {
      TF_RETURN_IF_ERROR(
          transposed_shape.AddDimWithStatus(input.dim_size(permutation[i])));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
      return CopyFrom(input, transposed_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, transposed_shape, output));
    mTensor input_mt = CreateMTensor(input);
    mTensor output_mt = CreateMTensor(*output);
    return TransposeFunctor::Compute(ctx, input_mt, permutation, output_mt);
  }

  // If there are repeated labels in either the input or output, then this
  // strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
  template <typename T>
  static Status StrideOrInflate(OpKernelContext* ctx, const Tensor& input,
                                const Labels& labels,
                                const LabelCounts& label_counts,
                                const bool should_inflate, Tensor* output) {
    // Return early if there are no repeated indices.
    if (absl::c_all_of(label_counts, [](int c) { return c <= 1; })) {
      return CopyFrom(input, input.shape(), output);
    }
    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    ShapeVec reshape;
    ShapeVec strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    ShapeVec strided_shape;
    ShapeVec inflated_shape;
    for (int label : labels) {
      const int count = label_counts[label];
      const int current_axis =
          should_inflate ? strided_shape.size() : inflated_shape.size();
      const int64_t dim = input.dim_size(current_axis);
      strided_shape.push_back(dim);
      inflated_shape.insert(inflated_shape.end(), count, dim);
      const int64_t reshape_dim = MathUtil::IPow(dim, count);
      reshape.push_back(reshape_dim);
      // While taking the d-diagonal in a rank k Tensor, we take d
      // equally-spaced elements including the first and last element. Then, (k
      // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
      const int64_t stride =
          (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
      strides.push_back(stride);
    }

    const ShapeVec& output_shape_dims =
        should_inflate ? inflated_shape : strided_shape;
    TensorShape output_shape;
    for (int64_t dim : output_shape_dims) {
      TF_RETURN_IF_ERROR(output_shape.AddDimWithStatus(dim));
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));

    const int rank = reshape.size();
    if (rank == 0) return OkStatus();

    auto compute_dense_strides = [](const ShapeVec& dims) {
      ShapeVec dense_strides(dims.size(), 1);
      for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
        dense_strides[i] = dense_strides[i + 1] * dims[i + 1];
      }
      return dense_strides;
    };

    const auto inflated_dense = compute_dense_strides(inflated_shape);
    ShapeVec diagonal_strides;
    diagonal_strides.reserve(rank);
    int inflated_axis = 0;
    for (int label : labels) {
      const int count = label_counts[label];
      int64_t diagonal_stride = 0;
      for (int i = 0; i < count; ++i) {
        diagonal_stride += inflated_dense[inflated_axis + i];
      }
      diagonal_strides.push_back(diagonal_stride);
      inflated_axis += count;
    }

    std::vector<int64_t> strided_dims_vec(strided_shape.begin(),
                                          strided_shape.end());
    std::vector<int64_t> diagonal_strides_vec(diagonal_strides.begin(),
                                              diagonal_strides.end());

    if (QueryMusaKernelRuntimeView(ctx).mudnn_handle == nullptr) {
      return MusaMudnnHandleRequiredError();
    }
    auto& handle = GetHandleByCtx(ctx);
    auto input_mt = CreateMTensor(input);
    auto output_mt = CreateMTensor(*output);

    if (should_inflate) {
      SetZeroFunctor::Compute<T>(ctx, &output_mt);
      output_mt.SetNdInfo(rank, strided_dims_vec.data(),
                          diagonal_strides_vec.data());
    } else {
      input_mt.SetNdInfo(rank, strided_dims_vec.data(),
                         diagonal_strides_vec.data());
    }

    ::musa::dnn::Unary op;
    auto status = op.SetMode(::musa::dnn::Unary::Mode::IDENTITY);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("Einsum StrideOrInflate SetMode failed. Status: ",
                              static_cast<int>(status));
    }
    status = op.Run(handle, output_mt, input_mt);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("Einsum StrideOrInflate Run failed. Status: ",
                              static_cast<int>(status));
    }
    return OkStatus();
  }

  // Returns true if the input dimensions are already sorted in the order
  // [batch, contract, free, reduce]. Used to implement an optimization to avoid
  // an extra transpose and instead uses (adj_x and adj_y) in BatchMatMul.
  static bool ShouldSwapFreeAndContract(
      const Labels& labels,
      const std::vector<EinsumDimensionType>& label_types) {
    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    gtl::InlinedVector<int, 5> remap = {0, 1, 3, 2, 4};
    for (int i = 0; i + 1 < labels.size(); ++i) {
      const int dimtype_a = remap[label_types[labels[i]]];
      const int dimtype_b = remap[label_types[labels[i + 1]]];
      if (dimtype_a > dimtype_b ||
          (dimtype_a == dimtype_b && labels[i] > labels[i + 1])) {
        return false;
      }
    }
    return true;
  }

  // TODO: BMatMul seems to perform worse when the input use half precision.
  template <typename T>
  static Status BMatMul(OpKernelContext* ctx, const Tensor& lhs,
                        const Tensor& rhs, bool trans_a, bool trans_b,
                        Tensor* output) {
    const Tensor& in0 = lhs;
    const Tensor& in1 = rhs;

    int64 d0 = in0.dim_size(in0.dims() - 2);
    int64 d1 = in0.dim_size(in0.dims() - 1);
    int64 d2 = in1.dim_size(in1.dims() - 2);
    int64 d3 = in1.dim_size(in1.dims() - 1);

    int64 m = trans_a ? d1 : d0;
    int64 k = trans_a ? d0 : d1;
    int64 n = trans_b ? d2 : d3;
    int64 k_check = trans_b ? d3 : d2;

    if (output->NumElements() == 0) return OkStatus();

    if (QueryMusaKernelRuntimeView(ctx).mudnn_handle == nullptr) {
      return MusaMudnnHandleRequiredError();
    }
    static bool tf32_enabled_global = []() {
      const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
      if (tf32_env) {
        return std::atoi(tf32_env) != 0;
      }
      return true;
    }();
    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_global);

    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*output);

    auto FixToBatchFormat = [](mTensor& mt, const Tensor& t) {
      if (t.dims() == 2) {
        int64_t rows = t.dim_size(0);
        int64_t cols = t.dim_size(1);
        mt.SetNdInfo({1, rows, cols}, {rows * cols, cols, 1});
      }
    };

    ::musa::dnn::Status status;

    mBatchMatMul op;
    op.SetTranspose(trans_a, trans_b);
    op.SetAlpha(1.0);
    op.SetBeta(0.0);

    FixToBatchFormat(mt_a, in0);
    FixToBatchFormat(mt_b, in1);
    FixToBatchFormat(mt_out, *output);

    status = op.Run(handle, mt_out, mt_a, mt_b);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA BatchMatMul execution failed. Status: ",
                              (int)status);
    }

    return OkStatus();
  }

  template <typename T>
  static Status AllocateBMatMulOutput(OpKernelContext* ctx, const Tensor& lhs,
                                      const Tensor& rhs, bool trans_a,
                                      bool trans_b, Tensor* output) {
    const int rank = lhs.dims();
    if (rank != rhs.dims() || rank < 2) {
      return errors::InvalidArgument("BatchMatMul expects matching rank >= 2");
    }
    TensorShape shape;
    for (int i = 0; i < rank - 2; ++i) {
      if (lhs.dim_size(i) != rhs.dim_size(i)) {
        return errors::InvalidArgument(
            "BatchMatMul fast path expects matching batch dimensions");
      }
      TF_RETURN_IF_ERROR(shape.AddDimWithStatus(lhs.dim_size(i)));
    }
    const int64_t m = trans_a ? lhs.dim_size(rank - 1) : lhs.dim_size(rank - 2);
    const int64_t k = trans_a ? lhs.dim_size(rank - 2) : lhs.dim_size(rank - 1);
    const int64_t n = trans_b ? rhs.dim_size(rank - 2) : rhs.dim_size(rank - 1);
    const int64_t k_check =
        trans_b ? rhs.dim_size(rank - 1) : rhs.dim_size(rank - 2);
    if (k != k_check) {
      return errors::InvalidArgument("BatchMatMul fast path contract mismatch");
    }
    TF_RETURN_IF_ERROR(shape.AddDimWithStatus(m));
    TF_RETURN_IF_ERROR(shape.AddDimWithStatus(n));
    return ctx->allocate_temp(DataTypeToEnum<T>::value, shape, output);
  }

  template <typename T>
  static Status ReduceOperand(
      OpKernelContext* ctx, const Tensor& input,
      const std::vector<EinsumDimensionType>& label_types,
      const LabelCounts& label_counts, Labels* labels, Labels* free_labels,
      bool* swap_free_and_contract, Tensor* output) {
    // Find the permutation to transpose the input dimensions in the order of
    // EinsumDimensionType; i.e. batch, free, contract and reduce dimensions.
    // This makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int64_t> permutation(input.dims());
    absl::c_iota(permutation, 0);
    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (ShouldSwapFreeAndContract(*labels, label_types)) {
      *swap_free_and_contract = true;
    } else {
      absl::c_sort(permutation, [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) <
               std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that EinsumDimensionTypes are in order.
    TF_RETURN_IF_ERROR(
        TransposeOperand<T>(ctx, input, permutation, &input_transposed));
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    TF_RETURN_IF_ERROR(
        StrideOrInflate<T>(ctx, input_transposed, *labels, label_counts,
                           false /* should_inflate */, &input_deduped));

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each EinsumDimensionType.
    gtl::InlinedVector<int64_t, 5> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
      const int label = labels->at(label_idx);
      int64_t dim = input_deduped.dim_size(label_idx);
      if (label_types[label] == EinsumDimensionType::kBroadcasting ||
          label_types[label] == EinsumDimensionType::kBatch) {
        TF_RETURN_IF_ERROR(output_shape.AddDimWithStatus(dim));
      } else if (label_types[label] == EinsumDimensionType::kFree) {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }
    if (*swap_free_and_contract)
      std::swap(reshape[EinsumDimensionType::kFree],
                reshape[EinsumDimensionType::kContract]);
    TF_RETURN_IF_ERROR(
        output_shape.AddDimWithStatus(reshape[EinsumDimensionType::kFree]));
    TF_RETURN_IF_ERROR(
        output_shape.AddDimWithStatus(reshape[EinsumDimensionType::kContract]));

    if (reshape[EinsumDimensionType::kReduce] == 1) {  // No need to reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    const int64_t reduce_size = reshape[kReduce];
    const int64_t output_size = reshape[kBroadcasting] * reshape[kBatch] *
                                reshape[kFree] * reshape[kContract];

    TensorShape input_flatten_shape;
    TF_RETURN_IF_ERROR(input_flatten_shape.AddDimWithStatus(output_size));
    TF_RETURN_IF_ERROR(input_flatten_shape.AddDimWithStatus(reduce_size));
    Tensor input_flattened;
    if (!input_flattened
             .BitcastFrom(input_deduped, input_deduped.dtype(),
                          input_flatten_shape)
             .ok()) {
      return errors::Internal("Failed to reshape Einsum input for reduce");
    }

    TensorShape output_flatten_shape;
    TF_RETURN_IF_ERROR(output_flatten_shape.AddDimWithStatus(output_size));
    Tensor output_flattened;
    if (!output_flattened
             .BitcastFrom(*output, output->dtype(), output_flatten_shape)
             .ok()) {
      return errors::Internal("Failed to reshape Einsum output for reduce");
    }

    auto input_mt = CreateMTensor(input_flattened);
    auto output_mt = CreateMTensor(output_flattened);

    int reduce_dims[] = {1};
    return ReduceFunctor::Compute<T>(
        ctx, &output_mt, &input_mt, ::musa::dnn::Reduce::Mode::ADD, reduce_dims,
        1, "MUSA Reduce (sum) execution failed. Status: ");
  }

  // Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
  static Status ReshapeToRank3(const Tensor& input, int batch_size,
                               Tensor* output) {
    const int rank = input.dims();
    TensorShape output_shape = {batch_size, input.dim_size(rank - 2),
                                input.dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
  }

  template <typename T>
  static Status MaterializeBroadcastedBatch(
      OpKernelContext* ctx, const Tensor& input,
      const TensorShape& output_batch_shape, Tensor* output) {
    const int input_rank = input.dims();
    if (input_rank < 2) {
      return errors::InvalidArgument(
          "Einsum batch broadcast expects rank >= 2, got rank ", input_rank);
    }

    const int input_batch_rank = input_rank - 2;
    const int output_batch_rank = output_batch_shape.dims();
    if (output_batch_rank < input_batch_rank) {
      return errors::Internal(
          "Einsum batch broadcast: output batch rank ", output_batch_rank,
          " is smaller than input batch rank ", input_batch_rank);
    }

    TensorShape output_shape = output_batch_shape;
    TF_RETURN_IF_ERROR(
        output_shape.AddDimWithStatus(input.dim_size(input_rank - 2)));
    TF_RETURN_IF_ERROR(
        output_shape.AddDimWithStatus(input.dim_size(input_rank - 1)));

    if (input.shape() == output_shape) {
      return CopyFrom(input, output_shape, output);
    }

    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    if (output->NumElements() == 0) return OkStatus();

    std::vector<int64_t> input_dense_strides(input_rank, 1);
    for (int axis = input_rank - 2; axis >= 0; --axis) {
      input_dense_strides[axis] =
          input_dense_strides[axis + 1] * input.dim_size(axis + 1);
    }

    const int target_rank = output_batch_rank + 2;
    std::vector<int64_t> target_dims(target_rank, 1);
    std::vector<int64_t> target_strides(target_rank, 0);
    const int batch_axis_offset = output_batch_rank - input_batch_rank;

    for (int out_axis = 0; out_axis < output_batch_rank; ++out_axis) {
      const int64_t out_dim = output_batch_shape.dim_size(out_axis);
      target_dims[out_axis] = out_dim;

      const int in_axis = out_axis - batch_axis_offset;
      if (in_axis < 0) {
        target_strides[out_axis] = 0;
        continue;
      }

      const int64_t in_dim = input.dim_size(in_axis);
      if (in_dim != out_dim && in_dim != 1) {
        return errors::Internal(
            "Einsum batch broadcast: incompatible batch dim at axis ", out_axis,
            ", input dim ", in_dim, ", output dim ", out_dim);
      }
      target_strides[out_axis] =
          (in_dim == 1 && out_dim != 1) ? 0 : input_dense_strides[in_axis];
    }

    target_dims[output_batch_rank] = input.dim_size(input_rank - 2);
    target_dims[output_batch_rank + 1] = input.dim_size(input_rank - 1);
    target_strides[output_batch_rank] = input_dense_strides[input_rank - 2];
    target_strides[output_batch_rank + 1] = input_dense_strides[input_rank - 1];

    auto input_mt = CreateMTensor(input);
    auto output_mt = CreateMTensor(*output);
    auto status = input_mt.SetNdInfo(target_rank, target_dims.data(),
                                     target_strides.data());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "Einsum batch broadcast: SetNdInfo failed. Status: ",
          static_cast<int>(status));
    }

    if (QueryMusaKernelRuntimeView(ctx).mudnn_handle == nullptr) {
      return MusaMudnnHandleRequiredError();
    }
    auto& handle = GetHandleByCtx(ctx);
    ::musa::dnn::Unary op;
    status = op.SetMode(::musa::dnn::Unary::Mode::IDENTITY);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "Einsum batch broadcast: Unary SetMode failed. Status: ",
          static_cast<int>(status));
    }
    status = op.Run(handle, output_mt, input_mt);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(
          "Einsum batch broadcast: Unary Run failed. Status: ",
          static_cast<int>(status));
    }
    return OkStatus();
  }

  // Contracts the inputs along the last axis (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  template <typename T>
  static Status ContractOperands(OpKernelContext* ctx,
                                 absl::Span<const Tensor> inputs,
                                 absl::Span<const bool> swap_free_and_contract,
                                 Tensor* output) {
    if (inputs.size() == 1)
      return CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());
    if (!bcast.IsValid()) {
      return errors::InvalidArgument(
          "Invalid broadcasting dimensions: ", inputs[0].shape().DebugString(),
          " vs. ", inputs[1].shape().DebugString());
    }

    const TensorShape output_batch_shape = bcast.output_batch_shape();
    Tensor lhs_broadcasted;
    TF_RETURN_IF_ERROR(MaterializeBroadcastedBatch<T>(
        ctx, inputs[0], output_batch_shape, &lhs_broadcasted));
    Tensor rhs_broadcasted;
    TF_RETURN_IF_ERROR(MaterializeBroadcastedBatch<T>(
        ctx, inputs[1], output_batch_shape, &rhs_broadcasted));

    Tensor lhs;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(lhs_broadcasted, bcast.output_batch_size(), &lhs));
    Tensor rhs;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(rhs_broadcasted, bcast.output_batch_size(), &rhs));

    TensorShape output_shape = output_batch_shape;
    for (int i = 0; i < inputs.size(); ++i) {
      const int64_t free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      TF_RETURN_IF_ERROR(
          output_shape.AddDimWithStatus(inputs[i].dim_size(free_axis)));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      mTensor output_mt = CreateMTensor(*output);
      TF_RETURN_IF_ERROR(SetZeroFunctor::Compute<T>(ctx, &output_mt));
      return OkStatus();
    }
    Tensor output_reshaped;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped));
    return BMatMul<T>(ctx, lhs, rhs, trans_x, trans_y, &output_reshaped);
  }
};

template <typename T>
class MusaEinsumOp : public MusaOpKernel {
 public:
  explicit MusaEinsumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equation", &equation_));
    OP_REQUIRES_OK(
        ctx, ParseEinsumEquation(equation_, &input_labels_, &output_labels_,
                                 &label_types_, &input_label_counts_,
                                 &output_label_counts_, &input_has_ellipsis_,
                                 &output_has_ellipsis_));
  }

  void Compute(OpKernelContext* ctx) override {
    bool fast_path_done = false;
    OP_REQUIRES_OK(ctx, TryComputeOneTrans3DEinsum(ctx, &fast_path_done));
    if (fast_path_done) {
      return;
    }

    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(input_labels_);
    Labels output_labels(output_labels_);
    std::vector<EinsumDimensionType> label_types(label_types_);
    OperandLabelCounts input_label_counts(input_label_counts_);
    LabelCounts output_label_counts(output_label_counts_);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(ctx, EinsumHelper::ProcessDimensions(
                            inputs, input_has_ellipsis_, output_has_ellipsis_,
                            &input_labels, &output_labels, &label_types,
                            &input_label_counts, &output_label_counts,
                            &label_to_dim_sizes));

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    // [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, EinsumHelper::ContractOperands<T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped));

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);

    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        OP_REQUIRES_OK(
            ctx, result_shape.AddDimWithStatus(label_to_dim_sizes[label]));
      }
    }

    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));

    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int64_t> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }
    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }

#if TF_MAJOR_VERSION < 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 10)
  string TraceString(const OpKernelContext& ctx, bool verbose) const override {
    string op = profiler::TraceMeOp(name_view(), type_string_view());
    string equation = strings::StrCat("(", equation_, ")");
    if (verbose) {
      string shape = ShapeTraceString(ctx);
      if (!shape.empty()) {
        return profiler::TraceMeEncode(
            std::move(op), {{"equation", equation}, {"shape", shape}});
      }
    }
    return profiler::TraceMeEncode(std::move(op), {{"equation", equation}});
  }
#endif

  // einsum contains bmm
  bool IsExpensive() override { return true; }

 private:
  Status TryComputeOneTrans3DEinsum(OpKernelContext* ctx, bool* done) {
    *done = false;
    if (std::getenv("MUSA_DISABLE_ONETRANS_3D_EINSUM_FASTPATH") != nullptr) {
      return ::tensorflow::OkStatus();
    }
    if (ctx->num_inputs() != 2) {
      return ::tensorflow::OkStatus();
    }

    const Tensor& raw_a = ctx->input(0);
    const Tensor& raw_b = ctx->input(1);
    if (raw_a.dims() != 3 || raw_b.dims() != 3) {
      return ::tensorflow::OkStatus();
    }

    const DataType target_dtype = DataTypeToEnum<T>::value;
    if (target_dtype != DT_FLOAT && target_dtype != DT_DOUBLE &&
        target_dtype != DT_HALF && target_dtype != DT_BFLOAT16) {
      return ::tensorflow::OkStatus();
    }

    Tensor a_tmp;
    Tensor b_tmp;
    const Tensor* a_ptr = &raw_a;
    const Tensor* b_ptr = &raw_b;
    if (target_dtype == DT_BFLOAT16 && raw_a.dtype() == DT_FLOAT) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_BFLOAT16, raw_a.shape(), &a_tmp));
      LaunchFloatToBFloat16Copy(
          raw_a.flat<float>().data(),
          reinterpret_cast<void*>(a_tmp.flat<bfloat16>().data()),
          static_cast<int64_t>(raw_a.NumElements()), GetMusaStreamByCtx(ctx));
      a_ptr = &a_tmp;
    } else if (raw_a.dtype() != target_dtype) {
      return ::tensorflow::OkStatus();
    }
    if (target_dtype == DT_BFLOAT16 && raw_b.dtype() == DT_FLOAT) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_BFLOAT16, raw_b.shape(), &b_tmp));
      LaunchFloatToBFloat16Copy(
          raw_b.flat<float>().data(),
          reinterpret_cast<void*>(b_tmp.flat<bfloat16>().data()),
          static_cast<int64_t>(raw_b.NumElements()), GetMusaStreamByCtx(ctx));
      b_ptr = &b_tmp;
    } else if (raw_b.dtype() != target_dtype) {
      return ::tensorflow::OkStatus();
    }

    const Tensor& a = *a_ptr;
    const Tensor& b = *b_ptr;
    if (equation_ == "btd,tde->bte") {
      const int64_t batch = a.dim_size(0);
      const int64_t tokens = a.dim_size(1);
      const int64_t depth = a.dim_size(2);
      if (b.dim_size(0) != tokens || b.dim_size(1) != depth) {
        return errors::InvalidArgument("Invalid btd,tde->bte input shapes: ",
                                       a.shape().DebugString(), " vs ",
                                       b.shape().DebugString());
      }
      Tensor lhs;
      TF_RETURN_IF_ERROR(
          EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &lhs));
      Tensor contracted;
      TF_RETURN_IF_ERROR(EinsumHelper::AllocateBMatMulOutput<T>(
          ctx, lhs, b, /*trans_a=*/false, /*trans_b=*/false, &contracted));
      TF_RETURN_IF_ERROR(EinsumHelper::BMatMul<T>(
          ctx, lhs, b, /*trans_a=*/false, /*trans_b=*/false, &contracted));
      Tensor output;
      TF_RETURN_IF_ERROR(EinsumHelper::TransposeOperand<T>(
          ctx, contracted, {1, 0, 2}, &output));
      ctx->set_output(0, output);
      *done = true;
      return ::tensorflow::OkStatus();
    }

    if (equation_ == "bte,tde->btd") {
      const int64_t batch = a.dim_size(0);
      const int64_t tokens = a.dim_size(1);
      const int64_t embed = a.dim_size(2);
      if (b.dim_size(0) != tokens || b.dim_size(2) != embed) {
        return errors::InvalidArgument("Invalid bte,tde->btd input shapes: ",
                                       a.shape().DebugString(), " vs ",
                                       b.shape().DebugString());
      }
      Tensor lhs;
      TF_RETURN_IF_ERROR(
          EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &lhs));
      Tensor contracted;
      TF_RETURN_IF_ERROR(EinsumHelper::AllocateBMatMulOutput<T>(
          ctx, lhs, b, /*trans_a=*/false, /*trans_b=*/true, &contracted));
      TF_RETURN_IF_ERROR(EinsumHelper::BMatMul<T>(
          ctx, lhs, b, /*trans_a=*/false, /*trans_b=*/true, &contracted));
      Tensor output;
      TF_RETURN_IF_ERROR(EinsumHelper::TransposeOperand<T>(
          ctx, contracted, {1, 0, 2}, &output));
      ctx->set_output(0, output);
      *done = true;
      return ::tensorflow::OkStatus();
    }

    if (equation_ == "bte,btd->tde") {
      const int64_t batch = a.dim_size(0);
      const int64_t tokens = a.dim_size(1);
      const int64_t embed = a.dim_size(2);
      if (b.dim_size(0) != batch || b.dim_size(1) != tokens) {
        return errors::InvalidArgument("Invalid bte,btd->tde input shapes: ",
                                       a.shape().DebugString(), " vs ",
                                       b.shape().DebugString());
      }
      Tensor lhs;
      TF_RETURN_IF_ERROR(
          EinsumHelper::TransposeOperand<T>(ctx, b, {1, 2, 0}, &lhs));
      Tensor rhs;
      TF_RETURN_IF_ERROR(
          EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &rhs));
      Tensor output;
      TF_RETURN_IF_ERROR(EinsumHelper::AllocateBMatMulOutput<T>(
          ctx, lhs, rhs, /*trans_a=*/false, /*trans_b=*/false, &output));
      TF_RETURN_IF_ERROR(EinsumHelper::BMatMul<T>(
          ctx, lhs, rhs, /*trans_a=*/false, /*trans_b=*/false, &output));
      ctx->set_output(0, output);
      *done = true;
      return ::tensorflow::OkStatus();
    }

    return ::tensorflow::OkStatus();
  }

  string equation_;
  OperandLabels input_labels_;
  Labels output_labels_;
  std::vector<EinsumDimensionType> label_types_;
  OperandLabelCounts input_label_counts_;
  LabelCounts output_label_counts_;
  gtl::InlinedVector<bool, 2> input_has_ellipsis_;
  bool output_has_ellipsis_ = false;
};  // class MusaEinsumOp

REGISTER_OP("MusaOneTrans3DEinsum")
    .Input("a: T")
    .Input("b: T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("equation: string");

template <typename T>
class MusaOneTrans3DEinsumOp : public MusaOpKernel {
 public:
  explicit MusaOneTrans3DEinsumOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equation", &equation_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    OP_REQUIRES(ctx, a.dims() == 3 && b.dims() == 3,
                errors::InvalidArgument(
                    "MusaOneTrans3DEinsum expects rank-3 inputs"));
    if (equation_ == "btd,tde->bte") {
      OP_REQUIRES(ctx, b.dim_size(0) == a.dim_size(1) &&
                           b.dim_size(1) == a.dim_size(2),
                  errors::InvalidArgument("Invalid btd,tde->bte shapes"));
      if (DataTypeToEnum<T>::value == DT_BFLOAT16) {
        const int64_t batch = a.dim_size(0);
        const int64_t tokens = a.dim_size(1);
        const int64_t depth = a.dim_size(2);
        const int64_t embed = b.dim_size(2);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
                                0, TensorShape({batch, tokens, embed}),
                                &output));
        OP_REQUIRES_OK(ctx, BFloat16StridedBatchedGemm(
                                ctx, b, a, output, MUBLAS_OP_N, MUBLAS_OP_N,
                                /*m=*/embed, /*n=*/batch, /*k=*/depth,
                                /*lda=*/embed, /*stride_a=*/depth * embed,
                                /*ldb=*/tokens * depth, /*stride_b=*/depth,
                                /*ldc=*/tokens * embed, /*stride_c=*/embed,
                                /*batch_count=*/tokens));
        return;
      }
      Tensor lhs;
      OP_REQUIRES_OK(
          ctx, EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &lhs));
      Tensor contracted;
      OP_REQUIRES_OK(ctx, EinsumHelper::AllocateBMatMulOutput<T>(
                              ctx, lhs, b, false, false, &contracted));
      OP_REQUIRES_OK(ctx, EinsumHelper::BMatMul<T>(
                              ctx, lhs, b, false, false, &contracted));
      Tensor output;
      OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<T>(
                              ctx, contracted, {1, 0, 2}, &output));
      ctx->set_output(0, output);
      return;
    }

    if (equation_ == "bte,tde->btd") {
      OP_REQUIRES(ctx, b.dim_size(0) == a.dim_size(1) &&
                           b.dim_size(2) == a.dim_size(2),
                  errors::InvalidArgument("Invalid bte,tde->btd shapes"));
      if (DataTypeToEnum<T>::value == DT_BFLOAT16) {
        const int64_t batch = a.dim_size(0);
        const int64_t tokens = a.dim_size(1);
        const int64_t embed = a.dim_size(2);
        const int64_t depth = b.dim_size(1);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
                                0, TensorShape({batch, tokens, depth}),
                                &output));
        OP_REQUIRES_OK(ctx, BFloat16StridedBatchedGemm(
                                ctx, b, a, output, MUBLAS_OP_T, MUBLAS_OP_N,
                                /*m=*/depth, /*n=*/batch, /*k=*/embed,
                                /*lda=*/embed, /*stride_a=*/depth * embed,
                                /*ldb=*/tokens * embed, /*stride_b=*/embed,
                                /*ldc=*/tokens * depth, /*stride_c=*/depth,
                                /*batch_count=*/tokens));
        return;
      }
      Tensor lhs;
      OP_REQUIRES_OK(
          ctx, EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &lhs));
      Tensor contracted;
      OP_REQUIRES_OK(ctx, EinsumHelper::AllocateBMatMulOutput<T>(
                              ctx, lhs, b, false, true, &contracted));
      OP_REQUIRES_OK(ctx, EinsumHelper::BMatMul<T>(
                              ctx, lhs, b, false, true, &contracted));
      Tensor output;
      OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<T>(
                              ctx, contracted, {1, 0, 2}, &output));
      ctx->set_output(0, output);
      return;
    }

    if (equation_ == "bte,btd->tde") {
      OP_REQUIRES(ctx, b.dim_size(0) == a.dim_size(0) &&
                           b.dim_size(1) == a.dim_size(1),
                  errors::InvalidArgument("Invalid bte,btd->tde shapes"));
      if (DataTypeToEnum<T>::value == DT_BFLOAT16) {
        const int64_t batch = a.dim_size(0);
        const int64_t tokens = a.dim_size(1);
        const int64_t embed = a.dim_size(2);
        const int64_t depth = b.dim_size(2);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
                                0, TensorShape({tokens, depth, embed}),
                                &output));
        OP_REQUIRES_OK(ctx, BFloat16StridedBatchedGemm(
                                ctx, a, b, output, MUBLAS_OP_N, MUBLAS_OP_T,
                                /*m=*/embed, /*n=*/depth, /*k=*/batch,
                                /*lda=*/tokens * embed, /*stride_a=*/embed,
                                /*ldb=*/tokens * depth, /*stride_b=*/depth,
                                /*ldc=*/embed, /*stride_c=*/depth * embed,
                                /*batch_count=*/tokens));
        return;
      }
      Tensor lhs;
      OP_REQUIRES_OK(
          ctx, EinsumHelper::TransposeOperand<T>(ctx, b, {1, 2, 0}, &lhs));
      Tensor rhs;
      OP_REQUIRES_OK(
          ctx, EinsumHelper::TransposeOperand<T>(ctx, a, {1, 0, 2}, &rhs));
      Tensor output;
      OP_REQUIRES_OK(ctx, EinsumHelper::AllocateBMatMulOutput<T>(
                              ctx, lhs, rhs, false, false, &output));
      OP_REQUIRES_OK(ctx, EinsumHelper::BMatMul<T>(
                              ctx, lhs, rhs, false, false, &output));
      ctx->set_output(0, output);
      return;
    }

    OP_REQUIRES(ctx, false,
                errors::InvalidArgument("Unsupported MusaOneTrans3DEinsum "
                                        "equation: ",
                                        equation_));
  }

 private:
  static Status BFloat16StridedBatchedGemm(
      OpKernelContext* ctx, const Tensor& lhs, const Tensor& rhs,
      Tensor* output, mublasOperation_t trans_a, mublasOperation_t trans_b,
      int m, int n, int k, int lda, long long int stride_a, int ldb,
      long long int stride_b, int ldc, long long int stride_c,
      int batch_count) {
    if (output->NumElements() == 0) return OkStatus();
    MusaKernelRuntimeView runtime = QueryMusaKernelRuntimeView(ctx);
    if (runtime.mublas_handle == nullptr) {
      return errors::Unimplemented(
          "MusaOneTrans3DEinsum BF16 muBLAS path requires mublas_handle");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* lhs_data = lhs.flat<bfloat16>().data();
    const void* rhs_data = rhs.flat<bfloat16>().data();
    void* output_data = output->flat<bfloat16>().data();
    const mublasStatus_t status = mublasGemmStridedBatchedEx(
        runtime.mublas_handle, trans_a, trans_b, m, n, k, &alpha, lhs_data,
        MUSA_R_16BF, lda, stride_a, rhs_data, MUSA_R_16BF, ldb, stride_b,
        &beta, output_data, MUSA_R_16BF, ldc, stride_c, batch_count,
        MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return errors::Internal(
          "MusaOneTrans3DEinsum BF16 muBLAS strided batched GEMM failed. "
          "Status: ",
          static_cast<int>(status));
    }
    return OkStatus();
  }

  string equation_;
};

#define REGISTER_MUSA_EINSUM(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Einsum").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaEinsumOp<TYPE>);

REGISTER_MUSA_EINSUM(float);
REGISTER_MUSA_EINSUM(double);
REGISTER_MUSA_EINSUM(int32);
REGISTER_MUSA_EINSUM(int64);
REGISTER_MUSA_EINSUM(Eigen::half);
REGISTER_MUSA_EINSUM(bfloat16);

#undef REGISTER_MUSA_EINSUM

#define REGISTER_MUSA_ONETRANS_3D_EINSUM(TYPE)                         \
  REGISTER_KERNEL_BUILDER(Name("MusaOneTrans3DEinsum")                  \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaOneTrans3DEinsumOp<TYPE>);

REGISTER_MUSA_ONETRANS_3D_EINSUM(float);
REGISTER_MUSA_ONETRANS_3D_EINSUM(double);
REGISTER_MUSA_ONETRANS_3D_EINSUM(Eigen::half);
REGISTER_MUSA_ONETRANS_3D_EINSUM(bfloat16);

#undef REGISTER_MUSA_ONETRANS_3D_EINSUM

}  // namespace musa
}  // namespace tensorflow
