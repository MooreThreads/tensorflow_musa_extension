#include "musa_einsum_op.h"

#include <functional>
#include <memory>

#include "../mu/device/musa_device.h"
#include "../utils/musa_einsum_op_util.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "mu/device/musa_memcpy.h"
#include "musa_fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

void DoTranspose(OpKernelContext* ctx, const Tensor& input,
                 const std::vector<int64_t>& permutation, Tensor* output);

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
    return Status::OK();
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
      return Status::OK();
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
    return Status::OK();
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
    if (output->CopyFrom(input, shape)) return Status::OK();
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
    DoTranspose(ctx, input, permutation, output);
    return Status::OK();
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
    if (rank == 0) return Status::OK();

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

    auto& handle = GetHandleByCtx(ctx);
    auto input_mt = CreateMTensor(input);
    auto output_mt = CreateMTensor(*output);

    if (should_inflate) {
      SetZeroFunctor<T> set_zero;
      set_zero(ctx, output);
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
    return Status::OK();
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

  template <typename T>
  static Status BMatMul(OpKernelContext* ctx, Tensor& lhs, const Tensor& rhs,
                        bool trans_a, bool trans_b, Tensor* output) {
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

    if (k != k_check) {
      return errors::InvalidArgument(
          "Matrix size-incompatible: In[0] mismatch In[1]");
    }

    if (output->dims() < 2) {
      return errors::Internal(
          "Einsum output tensor rank must be at least 2, got ", output->dims());
    }
    if (output->dim_size(output->dims() - 2) != m ||
        output->dim_size(output->dims() - 1) != n) {
      return errors::Internal(
          "Einsum output tensor shape mismatch, expected tail [", m, ", ", n,
          "], got ", output->shape().DebugString());
    }

    if (output->NumElements() == 0) return Status::OK();

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(false);
    // Use TF32 setting if needed, but here we can just default or use env like
    // matmul op. Since this is a static method, we don't have access to member
    // tf32_enabled_. Let's check environment variable again or just assume
    // default precision. For now, let's keep it simple and consistent with
    // standard usage.

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

    if (in0.dims() == 2 && in1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(trans_a, trans_b);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);

      status = op.Run(handle, mt_out, mt_a, mt_b);
      if (status != ::musa::dnn::Status::SUCCESS) {
        return errors::Internal(
            "MUSA MatMul (2D High Precision) execution failed. Status: ",
            (int)status);
      }
    } else {
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
    }
    return Status::OK();
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

    auto& handle = GetHandleByCtx(ctx);
    mReduce op;
    op.SetMode(::musa::dnn::Reduce::Mode::ADD);
    int reduce_dims[] = {1};
    op.SetDim(1, reduce_dims);

    // ------- TODO: Not sure if this would work in MUSA env -------
    tensorflow::Allocator* tf_allocator =
        ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
    // -------------------------------------------------------------
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

    auto status = op.Run(handle, output_mt, input_mt, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal("MUSA Reduce (sum) execution failed. Status: ",
                              static_cast<int>(status));
    }
    return Status::OK();
  }

  // Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
  static Status ReshapeToRank3(const Tensor& input, int batch_size,
                               Tensor* output) {
    const int rank = input.dims();
    TensorShape output_shape = {batch_size, input.dim_size(rank - 2),
                                input.dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
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
    Tensor lhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[0], bcast.x_batch_size(), &lhs));
    Tensor rhs;
    TF_RETURN_IF_ERROR(ReshapeToRank3(inputs[1], bcast.y_batch_size(), &rhs));
    TensorShape output_shape = bcast.output_batch_shape();
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
      SetZeroFunctor<T> set_zero;
      set_zero(ctx, output);
      return Status::OK();
    }
    Tensor output_reshaped;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped));
    // LaunchBatchMatMul<Device, T>::Launch(ctx, lhs, rhs, /*adj_x=*/false,
    //                                      /*adj_y=*/false, trans_x, trans_y,
    //                                      /*grad_x=*/false, /*grad_y=*/false,
    //                                      bcast, &output_reshaped);
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

 private:
  string equation_;
  OperandLabels input_labels_;
  Labels output_labels_;
  std::vector<EinsumDimensionType> label_types_;
  OperandLabelCounts input_label_counts_;
  LabelCounts output_label_counts_;
  gtl::InlinedVector<bool, 2> input_has_ellipsis_;
  bool output_has_ellipsis_ = false;
};  // class MusaEinsumOp

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

}  // namespace musa
}  // namespace tensorflow
