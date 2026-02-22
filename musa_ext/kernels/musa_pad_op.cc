#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Tpadding>
class MusaPadOp : public MusaOpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    static const int kMinDims = 0;
    static const int kMaxDims = 8;
    // Only support up to 8D,
    // which aligns with original TensorFlow implementation.
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));

    // The second input must be a 2D tensor with shape [rank(input), 2].
    // For each dimension of the input tensor, the corresponding row in the
    // paddings tensor specifies how many values to add before and after the
    // contents of that dimension, i.e. [pad_before, pad_after].
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    // Decide the value to pad with.
    T pad_value = T();
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      // The constant_values input must be a scalar.
      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(constant_values.shape()),
          errors::InvalidArgument("constant_values must be a scalar: ",
                                  constant_values.shape().DebugString()));
      pad_value = constant_values.scalar<T>()();
    }

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    typename TTypes<Tpadding>::ConstMatrix paddings = in1.matrix<Tpadding>();
    for (int d = 0; d < dims; ++d) {
      const Tpadding before_d =
          paddings(d, 0);                       // Pad before existing elements.
      const Tpadding after_d = paddings(d, 1);  // Pad after existing elements.
      OP_REQUIRES(context, before_d >= 0 && after_d >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before_d, " ", after_d));
      const int64_t size_d = in0.dim_size(d);
      OP_REQUIRES_OK(
          context, output_shape.AddDimWithStatus(before_d + size_d + after_d));
    }

    // If there is no padding to be done, forward the input to output.
    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      Tensor out;
      CHECK(out.CopyFrom(in0, output_shape));
      context->set_output(0, out);
      return;
    }

    TensorShape collapsed_input_shape;
    TensorShape collapsed_output_shape;
    Tensor collapsed_paddings;
    if (dims > 1 && CollapseAdjacentNonPaddedDimensions(
                        in0.shape(), in1, output_shape, &collapsed_input_shape,
                        &collapsed_paddings, &collapsed_output_shape)) {
      Tensor collapsed_input;
      CHECK(collapsed_input.CopyFrom(in0, collapsed_input_shape));
      Tensor collapsed_output;
      AllocatorAttributes alloc_attrs;
      alloc_attrs.set_on_host(context->input_memory_type(0) == HOST_MEMORY);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(collapsed_input.dtype(),
                                            collapsed_output_shape,
                                            &collapsed_output, alloc_attrs));
      const Tensor& collapsed_paddings_ref = collapsed_paddings;
      typename TTypes<Tpadding>::ConstMatrix collapsed_paddings_matrix =
          collapsed_paddings_ref.matrix<Tpadding>();

      OperateWithVariableRank(context, collapsed_input_shape.dims(),
                              collapsed_input, collapsed_paddings_matrix,
                              pad_value, &collapsed_output);

      Tensor output;
      CHECK(output.CopyFrom(collapsed_output, output_shape));
      context->set_output(0, output);
    } else {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      OperateWithVariableRank(context, dims, in0, paddings, pad_value, output);
    }
  }

 private:
  // Collapses adjacent dimensions that are not padded to one dimension for
  // speed. Returns true if any two dimensions are collapsed. For example,
  //
  //   Pad(input_shape=[8, 28, 28, 3],
  //       paddings=[[0, 0], [0, 0], [0, 0], [0, 1]]
  // is equivalent to
  //   Pad(input_shape=[6272, 3],
  //       paddings=[[0, 0], [0, 1]])
  //
  // input_shape: the original input shape.
  // paddings_as_tensor: the original paddings.
  // output_shape: the original output shape.
  // collapsed_input_shape: the input shape after collapsing.
  // collapsed_paddings_as_tensor: the paddings after collapsing.
  // collapsed_output_shape: the output shape after collapsing.
  static bool CollapseAdjacentNonPaddedDimensions(
      const TensorShape& input_shape, const Tensor& paddings_as_tensor,
      const TensorShape& output_shape, TensorShape* collapsed_input_shape,
      Tensor* collapsed_paddings_as_tensor,
      TensorShape* collapsed_output_shape) {
    bool collapsed = false;
    typename TTypes<Tpadding>::ConstMatrix paddings =
        paddings_as_tensor.matrix<Tpadding>();
    std::vector<std::pair<int, int>> collapsed_paddings;
    int i = 0;
    while (i < paddings.dimension(0)) {
      if (paddings(i, 0) != 0 || paddings(i, 1) != 0) {
        // If padded, copy the original dimension over.
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         input_shape.dim_size(i));
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          output_shape.dim_size(i));
        collapsed_paddings.push_back({paddings(i, 0), paddings(i, 1)});
        ++i;
      } else {
        // If not padded, find the next dimension that is padded and collapse
        // all dimensions in between to one dimension.
        int64_t collapsed_input_dim_size = input_shape.dim_size(i);
        int64_t collapsed_output_dim_size = output_shape.dim_size(i);
        ++i;
        while (i < paddings.dimension(0) && paddings(i, 0) == 0 &&
               paddings(i, 1) == 0) {
          collapsed = true;
          collapsed_input_dim_size *= input_shape.dim_size(i);
          collapsed_output_dim_size *= output_shape.dim_size(i);
          ++i;
        }
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         collapsed_input_dim_size);
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          collapsed_output_dim_size);
        collapsed_paddings.push_back({0, 0});
      }
    }

    // Copy collapsed_paddings to collapsed_paddings_as_tensor.
    *collapsed_paddings_as_tensor = Tensor(
        paddings_as_tensor.dtype(),
        TensorShape({static_cast<int64_t>(collapsed_paddings.size()), 2}));
    auto collapsed_paddings_as_matrix =
        collapsed_paddings_as_tensor->matrix<Tpadding>();
    for (size_t i = 0; i < collapsed_paddings.size(); ++i) {
      collapsed_paddings_as_matrix(i, 0) = collapsed_paddings[i].first;
      collapsed_paddings_as_matrix(i, 1) = collapsed_paddings[i].second;
    }
    return collapsed;
  }

  void OperateWithVariableRank(OpKernelContext* context, int fixed_dims,
                               const Tensor& input,
                               typename TTypes<Tpadding>::ConstMatrix paddings,
                               T pad_value, Tensor* output) {
    // Invoke the dims-specific implementation.
    switch (fixed_dims) {
      case 0:
        Operate<0>(context, input.tensor<T, 0>(), paddings, pad_value, output);
        break;
      case 1:
        // TODO(irving): Once Pad doesn't need a scalar special case,
        // change flat to tensor.  That is, once !allow_legacy_scalars().
        Operate<1>(context, input.flat<T>(), paddings, pad_value, output);
        break;
      case 2:
        Operate<2>(context, input.tensor<T, 2>(), paddings, pad_value, output);
        break;
      case 3:
        Operate<3>(context, input.tensor<T, 3>(), paddings, pad_value, output);
        break;
      case 4:
        Operate<4>(context, input.tensor<T, 4>(), paddings, pad_value, output);
        break;
      case 5:
        Operate<5>(context, input.tensor<T, 5>(), paddings, pad_value, output);
        break;
      case 6:
        Operate<6>(context, input.tensor<T, 6>(), paddings, pad_value, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 6 supported: ",
                                            input.shape().DebugString()));
    }
  }

  template <int Dims>
  void Operate(OpKernelContext* context,
               typename TTypes<T, Dims>::ConstTensor input,
               typename TTypes<Tpadding>::ConstMatrix paddings, T pad_value,
               Tensor* output) {
    CHECK_EQ(Dims, paddings.dimension(0));
    CHECK_EQ(2, paddings.dimension(1));
    Eigen::array<Eigen::IndexPair<Tpadding>, Dims> paddings_array;
    for (int i = 0; i < Dims; ++i) {
      paddings_array[i] = {paddings(i, 0), paddings(i, 1)};
    }
    functor::Pad<Device, T, Tpadding, Dims> functor;
    functor(context->eigen_device<Device>(), output->tensor<T, Dims>(), input,
            paddings_array, pad_value);
  }
}; // class MusaPadOp

}  // namespace musa
}  // namespace tensorflow