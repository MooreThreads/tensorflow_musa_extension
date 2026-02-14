#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename T, typename... Rest>
struct is_any : std::false_type {};

template <typename T, typename First>
struct is_any<T, First> : std::is_same<T, First> {};

template <typename T, typename First, typename... Rest>
struct is_any<T, First, Rest...>
    : std::integral_constant<bool, std::is_same<T, First>::value ||
                                       is_any<T, Rest...>::value> {};

template <typename T>
Status MusaFillCall(Tensor* out, T value, OpKernelContext* context) {
  mFill op;
  mHandle& h = GetHandleByCtx(context);
  auto out_mt = CreateMTensor(*out);

  if (is_any<T, int8, int16, int, int64, uint8, uint16, uint32, uint64,
             bool>::value) {
    if (mStatus::SUCCESS != op.SetValue(static_cast<int64_t>(value))) {
      return errors::Internal("mtdnn set value (int) error!");
    }
  } else if (is_any<T, float, double, Eigen::half, Eigen::bfloat16>::value) {
    if (mStatus::SUCCESS != op.SetValue(static_cast<double>(value))) {
      return errors::Internal("mtdnn set value (float) error!");
    }
  } else {
    return errors::Unimplemented("Data type not supported in MTGPU Fill.");
  }

  if (mStatus::SUCCESS != op.Run(h, out_mt)) {
    return errors::Internal("mtdnn run op error!");
  }

  return Status::OK();
}

}  // namespace

template <typename T, typename Index>
class MusaFillOp : public MusaOpKernel {
 public:
  explicit MusaFillOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& Tdims = context->input(0);
    const Tensor& Tvalue = context->input(1);

    OP_REQUIRES(
        context,
        (TensorShapeUtils::IsVector(Tdims.shape()) ||
         TensorShapeUtils::IsScalar(Tdims.shape())),
        errors::InvalidArgument("dims must represent a vector, got shape ",
                                Tdims.shape().DebugString()));

    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(Tvalue.shape()) ||
            (TensorShapeUtils::IsVector(Tvalue.shape()) &&
             Tvalue.shape().dim_size(0) == 1),
        errors::InvalidArgument("value must represent a scalar, got shape ",
                                Tvalue.shape().DebugString()));

    auto dims_vec = Tdims.flat<Index>();
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                reinterpret_cast<const Index*>(dims_vec.data()),
                                dims_vec.size(), &shape));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));

    if (shape.num_elements() == 0) return;

    OP_REQUIRES_OK(
        context, MusaFillCall(out, static_cast<T*>(Tvalue.data())[0], context));
  }
};

#define REGISTER_FILL_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device("MUSA")                        \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int32>("index_type")   \
                              .HostMemory("dims")                    \
                              .HostMemory("value"),                  \
                          MusaFillOp<type, int32>);                  \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device("MUSA")                        \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int64_t>("index_type") \
                              .HostMemory("dims")                    \
                              .HostMemory("value"),                  \
                          MusaFillOp<type, int64>);

REGISTER_FILL_KERNEL(float);
REGISTER_FILL_KERNEL(double);
REGISTER_FILL_KERNEL(int32);
REGISTER_FILL_KERNEL(int64);
REGISTER_FILL_KERNEL(Eigen::half);
REGISTER_FILL_KERNEL(Eigen::bfloat16);
REGISTER_FILL_KERNEL(bool);

#undef REGISTER_FILL_KERNEL

}  // namespace musa
}  // namespace tensorflow
