#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace musa {

class MusaMergeV2CheckpointsOp : public OpKernel {
 public:
  explicit MusaMergeV2CheckpointsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("delete_old_dirs", &delete_old_dirs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefixes = context->input(0);
    const Tensor& destination = context->input(1);

    auto prefixes_flat = prefixes.flat<tstring>();
    std::vector<tstring> prefixes_vec;
    for (int i = 0; i < prefixes_flat.size(); ++i)
      prefixes_vec.push_back(prefixes_flat(i));
    const tstring& dest_prefix = destination.flat<tstring>()(0);

    OP_REQUIRES_OK(context,
                   MergeBundles(context->env(), prefixes_vec, dest_prefix));
  }

 private:
  bool delete_old_dirs_;
};

REGISTER_KERNEL_BUILDER(Name("MergeV2Checkpoints")
                            .Device("MUSA")
                            .HostMemory("checkpoint_prefixes")
                            .HostMemory("destination_prefix"),
                        MusaMergeV2CheckpointsOp);

}  // namespace musa
}  // namespace tensorflow
