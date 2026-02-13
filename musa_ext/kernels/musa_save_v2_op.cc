#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace musa {

class MusaSaveV2Op : public OpKernel {
 public:
  explicit MusaSaveV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& prefix = ctx->input(0);
    const Tensor& tensor_names = ctx->input(1);
    const Tensor& shape_and_slices = ctx->input(2);

    const int kTensorsInputStart = 3;
    int num_tensors = ctx->num_inputs() - kTensorsInputStart;

    OP_REQUIRES(ctx, prefix.NumElements() == 1,
                errors::InvalidArgument("prefix must have 1 element"));
    OP_REQUIRES(ctx, tensor_names.NumElements() == num_tensors,
                errors::InvalidArgument("tensor_names must have ", num_tensors,
                                        " elements"));
    OP_REQUIRES(ctx, shape_and_slices.NumElements() == num_tensors,
                errors::InvalidArgument("shape_and_slices must have ",
                                        num_tensors, " elements"));

    const string& prefix_str = prefix.flat<tstring>()(0);
    const auto& names_flat = tensor_names.flat<tstring>();
    const auto& slices_flat = shape_and_slices.flat<tstring>();

    BundleWriter writer(ctx->env(), prefix_str);

    for (int i = 0; i < num_tensors; ++i) {
      const string& name = names_flat(i);
      const string& slice_spec = slices_flat(i);
      const Tensor& input_tensor = ctx->input(kTensorsInputStart + i);

      Status s;
      if (slice_spec.empty()) {
        s = writer.Add(name, input_tensor);
      } else {
        s = writer.AddSlice(name, input_tensor.shape(),
                            TensorSlice::ParseOrDie(slice_spec), input_tensor);
      }
      OP_REQUIRES_OK(ctx, s);
    }

    OP_REQUIRES_OK(ctx, writer.Finish());
  }
};

REGISTER_KERNEL_BUILDER(Name("SaveV2")
                            .Device("MUSA")
                            .HostMemory("prefix")
                            .HostMemory("tensor_names")
                            .HostMemory("shape_and_slices")
                            .HostMemory("tensors"),
                        MusaSaveV2Op);

}  // namespace musa
}  // namespace tensorflow
