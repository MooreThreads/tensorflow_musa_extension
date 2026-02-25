#include "mu/device/musa_device.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSwitchOp : public MusaOpKernel {
 public:
  explicit MusaSwitchOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& pred = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(pred.shape()),
                errors::InvalidArgument("pred must be a scalar: ",
                                        pred.shape().DebugString()));

    const bool pred_value = pred.scalar<bool>()();
    int port = (pred_value) ? 1 : 0;
    if (context->input_is_ref(0)) {
      context->forward_ref_input_to_ref_output(0, port);
    } else {
      context->set_output(port, context->input(0));
    }
  }
};  // class MusaSwitchOp

#define REGISTER_MUSA_SWITCH(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MusaSwitch").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaSwitchOp<type>);

REGISTER_MUSA_SWITCH(float);
REGISTER_MUSA_SWITCH(int32);
REGISTER_MUSA_SWITCH(int64);
REGISTER_MUSA_SWITCH(Eigen::half);
REGISTER_MUSA_SWITCH(bfloat16);
REGISTER_MUSA_SWITCH(double);
REGISTER_MUSA_SWITCH(uint8);
REGISTER_MUSA_SWITCH(bool);

}  // namespace musa
}  // namespace tensorflow