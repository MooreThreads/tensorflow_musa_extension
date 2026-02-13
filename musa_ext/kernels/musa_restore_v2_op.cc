#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace musa {

class MusaRestoreV2Op : public OpKernel {
 public:
  explicit MusaRestoreV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& prefix = ctx->input(0);
    const Tensor& tensor_names = ctx->input(1);
    const Tensor& shape_and_slices = ctx->input(2);

    OP_REQUIRES(ctx, prefix.NumElements() == 1,
                errors::InvalidArgument("prefix must have 1 element"));
    const string& prefix_str = prefix.flat<tstring>()(0);

    const auto& names_flat = tensor_names.flat<tstring>();
    const auto& slices_flat = shape_and_slices.flat<tstring>();

    int num_tensors = tensor_names.NumElements();
    OP_REQUIRES(ctx, shape_and_slices.NumElements() == num_tensors,
                errors::InvalidArgument(
                    "shape_and_slices must match tensor_names size"));
    OP_REQUIRES(ctx, dtypes_.size() == num_tensors,
                errors::InvalidArgument("dtypes must match tensor_names size"));

    BundleReader reader(ctx->env(), prefix_str);
    OP_REQUIRES_OK(ctx, reader.status());

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream != nullptr,
                errors::Internal("No MUSA stream available"));

    for (int i = 0; i < num_tensors; ++i) {
      const string& name = names_flat(i);
      const string& slice_spec = slices_flat(i);
      DataType dtype = dtypes_[i];

      TensorShape full_shape;
      OP_REQUIRES_OK(ctx, reader.LookupTensorShape(name, &full_shape));

      TensorShape target_shape;

      TensorSlice slice(full_shape.dims());
      bool is_slice = !slice_spec.empty();

      if (is_slice) {
        TensorShape parsed_shape;
        TensorSlice parsed_slice;
        TensorShape parsed_shape_slice;

        Status s = checkpoint::ParseShapeAndSlice(
            slice_spec, &parsed_shape, &parsed_slice, &parsed_shape_slice);

        if (s.ok()) {
          slice = parsed_slice;
          if (parsed_shape.dims() > 0 && parsed_shape != full_shape) {
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument(
                            "Shape in shape_and_slice spec does not match the "
                            "shape in the checkpoint file."));
          }
        } else {
          OP_REQUIRES_OK(ctx, TensorSlice::Parse(slice_spec, &slice));
        }

        OP_REQUIRES_OK(ctx, slice.SliceTensorShape(full_shape, &target_shape));
      } else {
        target_shape = full_shape;
      }

      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(i, target_shape, &output_tensor));

      if (target_shape.num_elements() == 0) continue;

      Tensor cpu_tensor;
      AllocatorAttributes cpu_alloc_attr;
      cpu_alloc_attr.set_on_host(true);
      cpu_alloc_attr.set_gpu_compatible(true);
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, target_shape, &cpu_tensor,
                                             cpu_alloc_attr));

      if (is_slice) {
        OP_REQUIRES_OK(ctx, reader.LookupSlice(name, slice, &cpu_tensor));
      } else {
        OP_REQUIRES_OK(ctx, reader.Lookup(name, &cpu_tensor));
      }

      const void* src_ptr = cpu_tensor.data();
      void* dst_ptr = output_tensor->data();
      uint64 copy_size_bytes =
          target_shape.num_elements() * DataTypeSize(dtype);

      se::DeviceMemoryBase dst_mem(dst_ptr, copy_size_bytes);
      stream->ThenMemcpy(&dst_mem, src_ptr, copy_size_bytes);

      OP_REQUIRES_OK(ctx, stream->BlockHostUntilDone());
    }
  }

 private:
  std::vector<DataType> dtypes_;
};

REGISTER_KERNEL_BUILDER(Name("RestoreV2")
                            .Device("MUSA")
                            .HostMemory("prefix")
                            .HostMemory("tensor_names")
                            .HostMemory("shape_and_slices"),
                        MusaRestoreV2Op);

}  // namespace musa
}  // namespace tensorflow