#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Index>
class MusaSparseSoftMaxCrossEntroyWithLogitsOp : public MusaOpKernel {
 public:
  explicit MusaSparseSoftMaxCrossEntroyWithLogitsOp(
      OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits = context->input(0);
    const Tensor& labels = context->input(1);

    OP_REQUIRES(context, logits.dims() == 2,
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(context, labels.dims() == 1,
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same batch dimension"));

    const int64 batch_size = logits.dim_size(0);
    const int64 num_classes = logits.dim_size(1);

    Tensor* loss_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, labels.shape(), &loss_tensor));
    Tensor* back_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, logits.shape(), &back_tensor));

    if (logits.NumElements() == 0) return;

    mHandle& h = GetHandleByCtx(context);

    Tensor log_probs_t, gathered_log_t, ones_t;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                logits.dtype(), logits.shape(), &log_probs_t));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(logits.dtype(), labels.shape(),
                                          &gathered_log_t));
    OP_REQUIRES_OK(context, context->allocate_temp(logits.dtype(),
                                                   labels.shape(), &ones_t));

    mTensor logits_m = CreateMTensor(logits, format_);
    mTensor labels_m = CreateMTensor(labels, format_);
    mTensor log_probs_m = CreateMTensor(log_probs_t, format_);
    mTensor gathered_log_m = CreateMTensor(gathered_log_t, format_);
    mTensor ones_m = CreateMTensor(ones_t, format_);
    mTensor loss_m = CreateMTensor(*loss_tensor, format_);
    mTensor back_m = CreateMTensor(*back_tensor, format_);

    mSoftmax softmax_op;
    softmax_op.SetAlgorithm(mSoftmax::Algorithm::ACCURATE);
    softmax_op.SetDim(1);

    softmax_op.SetMode(SOFTMAX_MODE::LOGSOFTMAX);
    softmax_op.Run(h, log_probs_m, logits_m);

    mGatherX gather_op;
    gather_op.SetMode(mGatherX::Mode::GATHER);
    gather_op.SetAxis(1);
    gather_op.SetBatchDims(1);
    gather_op.Run(h, gathered_log_m, labels_m, log_probs_m);

    mUnary unary_op;
    unary_op.SetMode(UNARY_MODE::MUL);
    unary_op.SetAlpha(-1.0);
    unary_op.Run(h, loss_m, gathered_log_m);

    softmax_op.SetMode(SOFTMAX_MODE::SOFTMAX);
    softmax_op.Run(h, back_m, logits_m);

    mFill fill_op;
    fill_op.SetValue(1.0);
    fill_op.Run(h, ones_m);

    std::vector<int64_t> fake_2d_shape = {batch_size, 1};
    labels_m.SetNdInfo(static_cast<int>(fake_2d_shape.size()),
                       fake_2d_shape.data());
    ones_m.SetNdInfo(static_cast<int>(fake_2d_shape.size()),
                     fake_2d_shape.data());

    mScatter scatter_op;
    scatter_op.SetMode(mScatter::Mode::SUB);
    scatter_op.Run(h, back_m, labels_m, ones_m, 1, nullptr);

    //  VLOG(1) << ">>> [MUSA_TRACE] SparseSoftmaxCrossEntropy computed
    //  successfully.";
  }
};

#define REGISTER_MUSA_SPARSE_XENT_COMBINE(T, Index)                   \
  REGISTER_KERNEL_BUILDER(Name("SparseSoftmaxCrossEntropyWithLogits") \
                              .Device(DEVICE_MTGPU)                   \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Index>("Tlabels"),      \
                          MusaSparseSoftMaxCrossEntroyWithLogitsOp<T, Index>);

#define REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(T) \
  REGISTER_MUSA_SPARSE_XENT_COMBINE(T, int32);  \
  REGISTER_MUSA_SPARSE_XENT_COMBINE(T, int64);

REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(float);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(double);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(Eigen::half);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(bfloat16);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(int32);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(int64);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(uint8);
REGISTER_MUSA_SPARSE_XENT_ALL_LABELS(bool);

}  // namespace musa
}  // namespace tensorflow
