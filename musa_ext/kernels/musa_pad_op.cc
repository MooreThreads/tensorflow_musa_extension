#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename ValueT>
std::enable_if_t<std::is_integral<ValueT>::value, mStatus> SetPadValue(
    mPad& pad, ValueT value) {
  return pad.SetValue(static_cast<int64_t>(value));
}

template <typename ValueT>
std::enable_if_t<!std::is_integral<ValueT>::value, mStatus> SetPadValue(
    mPad& pad, ValueT value) {
  return pad.SetValue(static_cast<double>(value));
}

}  // namespace

template <typename T, typename Tpadding>
class MusaPadOp : public MusaOpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& paddings = context->input(1);
    const int dims = input.dims();
    static constexpr int kMinDims = 0;
    static constexpr int kMaxDims = 8;

    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(paddings.shape()) &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                paddings.shape().DebugString()));
    OP_REQUIRES(context, dims == paddings.dim_size(0),
                errors::InvalidArgument(
                    "The first dimension of paddings must be the rank of inputs ",
                    paddings.shape().DebugString(), " ",
                    input.shape().DebugString()));

    T pad_value = T();
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(constant_values.shape()),
                  errors::InvalidArgument("constant_values must be a scalar: ",
                                          constant_values.shape().DebugString()));
      pad_value = constant_values.scalar<T>()();
    }

    TensorShape output_shape;
    typename TTypes<Tpadding>::ConstMatrix paddings_matrix =
        paddings.matrix<Tpadding>();
    std::vector<std::pair<int, int>> pad_pairs_per_dim;
    pad_pairs_per_dim.reserve(dims);
    for (int d = 0; d < dims; ++d) {
      const int64_t before = static_cast<int64_t>(paddings_matrix(d, 0));
      const int64_t after = static_cast<int64_t>(paddings_matrix(d, 1));
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before, " ", after));
      OP_REQUIRES(context,
                  before <= std::numeric_limits<int>::max() &&
                      after <= std::numeric_limits<int>::max(),
                  errors::InvalidArgument(
                      "Paddings must fit in int32 for MUSA: ", before, " ",
                      after));
      pad_pairs_per_dim.emplace_back(static_cast<int>(before),
                                     static_cast<int>(after));
      const int64_t size_d = input.dim_size(d);
      OP_REQUIRES_OK(
          context,
          output_shape.AddDimWithStatus(before + size_d + after));
    }

    // !!!
    // muDNN Pad expects padding pairs ordered from the innermost dimension
    // outward (reverse TensorFlow's dimension order).
    // !!!
    std::vector<int> pad_pairs;
    pad_pairs.reserve(dims * 2);
    for (int d = dims - 1; d >= 0; --d) {
      pad_pairs.push_back(pad_pairs_per_dim[d].first);
      pad_pairs.push_back(pad_pairs_per_dim[d].second);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);
    auto& handle = GetHandleByCtx(context);

    mPad pad_op;
    pad_op.SetMode(mPad::Mode::CONSTANT);
    SetPadValue(pad_op, pad_value);
    if (!pad_pairs.empty()) {
      pad_op.SetPaddingInfo(static_cast<int>(pad_pairs.size()),
                           pad_pairs.data());
    } else {
      pad_op.SetPaddingInfo(0, nullptr);
    }

    auto status = pad_op.Run(handle, out_mt, in_mt);
    OP_REQUIRES(context, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Pad execution failed."));
  }
};

#define REGISTER_MUSA_PAD_TYPE(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(Name("Pad")                                         \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          MusaPadOp<TYPE, int32>);                             \
  REGISTER_KERNEL_BUILDER(Name("Pad")                                         \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int64>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          MusaPadOp<TYPE, int64>);                             \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                                       \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings")                          \
                              .HostMemory("constant_values"),                  \
                          MusaPadOp<TYPE, int32>);                             \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                                       \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int64>("Tpaddings")              \
                              .HostMemory("paddings")                          \
                              .HostMemory("constant_values"),                  \
                          MusaPadOp<TYPE, int64>);

REGISTER_MUSA_PAD_TYPE(float);
REGISTER_MUSA_PAD_TYPE(int32);
REGISTER_MUSA_PAD_TYPE(int64);
REGISTER_MUSA_PAD_TYPE(Eigen::half);
REGISTER_MUSA_PAD_TYPE(bfloat16);
REGISTER_MUSA_PAD_TYPE(double);
REGISTER_MUSA_PAD_TYPE(uint8);

#undef REGISTER_MUSA_PAD_TYPE

}  // namespace musa
}  // namespace tensorflow
