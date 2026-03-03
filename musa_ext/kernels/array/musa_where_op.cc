#include "musa_where_op.h"

#include "../utils_op.h"
#include "tensorflow/core/util/cuda_solvers.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaWhereOp : public MusaOpKernel {
 public:
  explicit MusaWhereOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  template <typename Tindex>
  void Compute(OpKernelContext* context) override {
    ScratchShape<Tindex> num_true(context, 1,
                                  /*on_host=*/true);  // pinned memory
    typename TTypes<Tindex>::Unaligned num_true_t(num_true.mutable_data());
    const int input_dims = input.dims();

    Status s = NumTrue::Compute(context, input.flat<T>(), num_true_t);
    OP_REQUIRES_OK(context, s.status());

    auto create_and_check_output = [context, &input, input_dims,
                                    num_true = std::move(num_true)]() {
      musaStream_t stream = GetMusaStreamByCtx(ctx);
      {
        auto scoped_activation = stream->parent()->Activate();

        Tindex found_true = -1;
        Tensor* output = nullptr;
        OP_REQUIRES_OK(
            context,
            context->allocate_output(
                0, TensorShape({*num_true.data(), input_dims}), &output));

        // TODO: Replace with MUSA supported implementation
        const int NDIM = input_dims;
        Status s = Where::Compute<NDIM, T, Tindex>(
            context, input.tensor<T, NDIM>(), output->matrix<int64_t>(),
            &found_true);
        OP_REQUIRES_OK(context, s);
      }
    };
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, create_and_check_output);
  }

  // where op contains customied kernel
  bool IsExpensive() override { return true; }
};

}  // namespace musa
}  // namespace tensorflow