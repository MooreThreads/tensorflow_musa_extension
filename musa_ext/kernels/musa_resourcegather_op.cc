#include <mudnn.h>

#include <cmath>
#include <limits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/platform/logging.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename Index>
class MusaResourceGatherOp : public MusaOpKernel {
 public:
  explicit MusaResourceGatherOp(OpKernelConstruction* c) : MusaOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    Status s = LookupResource(c, HandleFromInput(c, 0), &v);
    if (!s.ok()) {
      c->CtxFailure(s);
      return;
    }

    tf_shared_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);

    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));
    OP_REQUIRES(c, params.shape().dims() >= batch_dims_,
                errors::InvalidArgument("params must have at least ",
                                        batch_dims_, " dims"));

    TensorShape result_shape;
    for (int i = 0; i < batch_dims_; ++i)
      result_shape.AddDim(params.dim_size(i));
    for (int i = batch_dims_; i < indices.dims(); ++i)
      result_shape.AddDim(indices.dim_size(i));
    for (int i = batch_dims_ + 1; i < params.dims(); ++i)
      result_shape.AddDim(params.dim_size(i));

    Tensor* out = nullptr;
    s = c->allocate_output(0, result_shape, &out);
    if (!s.ok()) {
      c->CtxFailure(s);
      return;
    }

    if (out->NumElements() == 0) {
      //  LOG(INFO) << ">>>>> [MUSA_DEBUG] ResourceGather: Output is empty,
      //  returning early.";
      return;
    }

    if (indices.NumElements() > 0) {
      auto& h = GetHandleByCtx(c);
      mGatherX op;

      MTOP_CHECK_OK(op.SetMode(mGatherX::Mode::GATHER), "SetMode", c);
      MTOP_CHECK_OK(op.SetAxis(static_cast<int>(batch_dims_)), "SetAxis", c);

      auto out_mt = CreateMTensor(*out, format_);
      auto indices_mt = CreateMTensor(indices, format_);
      auto params_mt = CreateMTensor(params, format_);

      auto status = op.Run(h, out_mt, indices_mt, params_mt);
      if (status != ::musa::dnn::Status::SUCCESS) {
        //  LOG(ERROR) << ">>>>> [MUSA_DEBUG] ResourceGather: mGatherX Run
        //  Failed! Status=" << static_cast<int>(status);
        c->CtxFailure(errors::Internal("mGatherX execution failed"));
        return;
      }
      //  LOG(INFO) << ">>>>> [MUSA_DEBUG] ResourceGather: mGatherX Success.";
    } else {
      //  LOG(INFO) << ">>>>> [MUSA_DEBUG] ResourceGather: Indices empty,
      //  skipping Kernel.";
    }
  }

 private:
  int32 batch_dims_ = 0;
};

template <typename T, typename Index>
class MusaResourceScatterAddOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    if (indices.NumElements() > 0) {
      auto& h = GetHandleByCtx(c);
      auto* device = static_cast<MusaDevice*>(c->device());
      auto maintainer = device->GetMemMaintainer(
          [](size_t s) { return ::musa::dnn::MemoryHandler(); });

      mScatterND op;
      MTOP_CHECK_OK(op.SetMode(mScatterND::Mode::ADD), "SetModeAdd", c);

      Tensor indices_reshaped;
      TensorShape indices_new_shape = indices.shape();
      indices_new_shape.AddDim(1);

      if (!indices_reshaped
               .BitcastFrom(indices, indices.dtype(), indices_new_shape)
               .ok()) {
        OP_REQUIRES(c, false,
                    errors::Internal(
                        "MusaResourceScatterAdd: Failed to reshape indices."));
      }

      auto params_mt = CreateMTensor(*params, format_);
      auto indices_mt = CreateMTensor(indices_reshaped, format_);
      auto updates_mt = CreateMTensor(updates, format_);
      MTOP_CHECK_OK_RUN(
          op.Run(h, params_mt, indices_mt, updates_mt, maintainer),
          "RunScatterND", c);
    }

    if (c->num_outputs() > 0) {
      c->set_output(0, c->input(0));
    }
  }
};

template <typename T, mBinary::Mode BMODE>
class MusaAssignUpdateVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &variable));
    mutex_lock ml(*variable->mu());

    Tensor* var_tensor = variable->tensor();
    const Tensor& value = c->input(1);

    if (var_tensor->NumElements() > 0) {
      auto& h = GetHandleByCtx(c);
      mBinary op;
      MTOP_CHECK_OK(op.SetMode(BMODE), "SetMode", c);
      auto out_mt = CreateMTensor(*var_tensor, format_);
      auto in_mt = CreateMTensor(value, format_);
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, out_mt, in_mt), "RunBinaryUpdate", c);
    }

    if (c->num_outputs() > 0) {
      c->set_output(0, c->input(0));
    }
  }
};

class MusaVariableShapeOp : public OpKernel {
 public:
  explicit MusaVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    tf_shared_lock ml(*v->mu());
    const TensorShape& s = v->tensor()->shape();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({s.dims()}), &out));
    for (int i = 0; i < s.dims(); ++i) {
      if (out->dtype() == DT_INT32)
        out->flat<int32>()(i) = s.dim_size(i);
      else
        out->flat<int64>()(i) = s.dim_size(i);
    }
  }
};

#define REGISTER_MUSA_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceGatherOp<type, int32>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceGatherOp<type, int64>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceScatterAddOp<type, int32>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceScatterAddOp<type, int64>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignSubVariableOp")                                 \
          .Device(DEVICE_MTGPU)                                   \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::SUB>);      \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignAddVariableOp")                                 \
          .Device(DEVICE_MTGPU)                                   \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::ADD>);

REGISTER_MUSA_KERNELS(float);
REGISTER_MUSA_KERNELS(Eigen::half);

REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("resource")
                            .TypeConstraint<int64>("dtype"),
                        MusaAssignUpdateVariableOp<int64, mBinary::Mode::ADD>);
REGISTER_KERNEL_BUILDER(Name("AssignSubVariableOp")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("resource")
                            .TypeConstraint<int64>("dtype"),
                        MusaAssignUpdateVariableOp<int64, mBinary::Mode::SUB>);
REGISTER_KERNEL_BUILDER(Name("AssignAddVariableOp")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("resource")
                            .TypeConstraint<int32>("dtype"),
                        MusaAssignUpdateVariableOp<int32, mBinary::Mode::ADD>);
REGISTER_KERNEL_BUILDER(Name("AssignSubVariableOp")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("resource")
                            .TypeConstraint<int32>("dtype"),
                        MusaAssignUpdateVariableOp<int32, mBinary::Mode::SUB>);

REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        MusaVariableShapeOp);

}  // namespace musa
}  // namespace tensorflow
