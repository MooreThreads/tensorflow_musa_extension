#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace musa {

class MusaSqueezeOp : public OpKernel {
 public:
  explicit MusaSqueezeOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("squeeze_dims", &squeeze_dims_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);

    TensorShape output_shape;
    for (int i = 0; i < input.dims(); ++i) {
      bool should_squeeze = false;
      if (input.dim_size(i) == 1) {
        if (squeeze_dims_.empty()) {
          should_squeeze = true;
        } else {
          for (int d : squeeze_dims_) {
            int positive_d = d < 0 ? d + input.dims() : d;
            if (i == positive_d) {
              should_squeeze = true;
              break;
            }
          }
        }
      }
      if (!should_squeeze) {
        output_shape.AddDim(input.dim_size(i));
      }
    }

    Tensor output;
    if (!output.CopyFrom(input, output_shape)) {
      c->CtxFailure(errors::Internal("Failed to squeeze tensor shape"));
      return;
    }

    c->set_output(0, output);
  }

 private:
  std::vector<int32> squeeze_dims_;
};

#define REGISTER_MUSA_SQUEEZE(type)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Squeeze").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSqueezeOp);

REGISTER_MUSA_SQUEEZE(float);
REGISTER_MUSA_SQUEEZE(Eigen::half);
REGISTER_MUSA_SQUEEZE(bfloat16);
REGISTER_MUSA_SQUEEZE(int32);
REGISTER_MUSA_SQUEEZE(int64);
REGISTER_MUSA_SQUEEZE(bool);
REGISTER_MUSA_SQUEEZE(double);
REGISTER_MUSA_SQUEEZE(uint8);

}  // namespace musa
}  // namespace tensorflow
