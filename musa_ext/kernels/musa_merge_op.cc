#include "mu/device/musa_device.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMergeOp : public MusaOpKernel {
 public:
  explicit MusaMergeOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    bool input_seen = false;
    for (int i = 0; i < context->num_inputs(); ++i) {
      if (context->has_input(i)) {
        if (input_seen) {
          LOG(WARNING) << "Merge op has more than one valid input. This "
                       << "indicates that the graph doesn't use merge op "
                       << "properly. Please check your graph. ";
          return;
        }
        input_seen = true;

        if (IsRefType(context->input_dtype(i))) {
          context->forward_ref_input_to_ref_output(i, 0);
        } else {
          context->set_output(0, context->input(i));
        }
        // The value_index output is typically used only in gradient
        // calculations, so we can avoid allocating in many inference workloads.
        if (context->output_required(1)) {
          Tensor* value_index = nullptr;
          OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                           &value_index));
          value_index->scalar<int32_t>()() = i;
        }
      }
    }
  }
};  // class MusaMergeOp

#define REGISTER_MUSA_MERGE(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Merge").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaMergeOp<type>);

REGISTER_MUSA_MERGE(float);
REGISTER_MUSA_MERGE(int32);
REGISTER_MUSA_MERGE(int64);
REGISTER_MUSA_MERGE(Eigen::half);
REGISTER_MUSA_MERGE(bfloat16);
REGISTER_MUSA_MERGE(double);
REGISTER_MUSA_MERGE(uint8);
REGISTER_MUSA_MERGE(bool);

}  // namespace musa
}  // namespace tensorflow