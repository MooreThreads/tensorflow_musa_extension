// Optimized MUSA ResourceGather Op Implementation
// Uses custom kernels for maximum performance
//
// Performance optimizations:
// 1. Custom MUSA kernels with optimized memory access patterns
// 2. GPU-side bounds checking
// 3. Direct kernel launch without muDNN overhead
// 4. Support for all data types including bfloat16 and double

#include <mudnn.h>
#include <musa_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/platform/logging.h"

// ============================================================================
// Custom Kernel Launcher Declarations
// ============================================================================

extern "C" {
void LaunchResourceGatherFloatInt32(const float* params, const int* indices,
                                    float* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherFloatInt64(const float* params, const int64_t* indices,
                                    float* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int64_t limit,
                                    musaStream_t stream);
void LaunchResourceGatherDoubleInt32(const double* params, const int* indices,
                                     double* output, int64_t batch_size,
                                     int64_t inner_size, int64_t indices_size,
                                     int64_t params_stride, int limit,
                                     musaStream_t stream);
void LaunchResourceGatherDoubleInt64(const double* params,
                                     const int64_t* indices, double* output,
                                     int64_t batch_size, int64_t inner_size,
                                     int64_t indices_size,
                                     int64_t params_stride, int64_t limit,
                                     musaStream_t stream);
void LaunchResourceGatherInt32Int32(const int* params, const int* indices,
                                    int* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt32Int64(const int* params, const int64_t* indices,
                                    int* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int64_t limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt64Int32(const int64_t* params, const int* indices,
                                    int64_t* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt64Int64(const int64_t* params,
                                    const int64_t* indices, int64_t* output,
                                    int64_t batch_size, int64_t inner_size,
                                    int64_t indices_size, int64_t params_stride,
                                    int64_t limit, musaStream_t stream);
void LaunchResourceGatherHalfInt32(const void* params, const int* indices,
                                   void* output, int64_t batch_size,
                                   int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, int limit,
                                   musaStream_t stream);
void LaunchResourceGatherHalfInt64(const void* params, const int64_t* indices,
                                   void* output, int64_t batch_size,
                                   int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, int64_t limit,
                                   musaStream_t stream);
void LaunchResourceScatterAddRowsFloatInt32(
    float* params, const int* indices, const float* updates,
    int64_t num_updates, int64_t slice_size, int limit, musaStream_t stream);
void LaunchResourceScatterAddRowsFloatInt64(
    float* params, const int64_t* indices, const float* updates,
    int64_t num_updates, int64_t slice_size, int64_t limit,
    musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ============================================================================
// Optimized ResourceGather Op Implementation
// ============================================================================

template <typename T, typename Index>
class MusaResourceGatherOp : public MusaOpKernel {
 public:
  explicit MusaResourceGatherOp(OpKernelConstruction* c) : MusaOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  bool IsExpensive() override { return true; }

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

    // Build output shape
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
      return;
    }

    if (indices.NumElements() > 0) {
      // Calculate dimensions for kernel launch
      int64_t batch_size = 1;
      for (int i = 0; i < batch_dims_; ++i) {
        batch_size *= params.dim_size(i);
      }

      int64_t inner_size = 1;
      for (int i = batch_dims_ + 1; i < params.dims(); ++i) {
        inner_size *= params.dim_size(i);
      }

      const int64_t indices_size = indices.NumElements();
      const int64_t params_stride = params.dim_size(batch_dims_) * inner_size;
      const Index limit = static_cast<Index>(params.dim_size(batch_dims_));

      musaStream_t stream = GetMusaStreamByCtx(c);
      OP_REQUIRES(
          c, stream != nullptr,
          errors::Internal("MUSA stream is unavailable for ResourceGather"));

      // Launch optimized kernel
      LaunchKernel(params.flat<T>().data(), indices.flat<Index>().data(),
                   out->flat<T>().data(), batch_size, inner_size, indices_size,
                   params_stride, limit, stream);
    }
  }

 private:
  int32 batch_dims_ = 0;

  void LaunchKernel(const T* params, const Index* indices, T* output,
                    int64_t batch_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, Index limit,
                    musaStream_t stream);
};

// ============================================================================
// Launcher Specializations
// ============================================================================

#define DEFINE_RESOURCE_GATHER_LAUNCHER(T, IndexT, launcher_func)            \
  template <>                                                                \
  void MusaResourceGatherOp<T, IndexT>::LaunchKernel(                        \
      const T* params, const IndexT* indices, T* output, int64_t batch_size, \
      int64_t inner_size, int64_t indices_size, int64_t params_stride,       \
      IndexT limit, musaStream_t stream) {                                   \
    launcher_func(params, indices, output, batch_size, inner_size,           \
                  indices_size, params_stride, limit, stream);               \
  }

DEFINE_RESOURCE_GATHER_LAUNCHER(float, int32, LaunchResourceGatherFloatInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(float, int64, LaunchResourceGatherFloatInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, int32, LaunchResourceGatherDoubleInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, int64, LaunchResourceGatherDoubleInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(int32, int32, LaunchResourceGatherInt32Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(int32, int64, LaunchResourceGatherInt32Int64)
DEFINE_RESOURCE_GATHER_LAUNCHER(int64, int32, LaunchResourceGatherInt64Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(int64, int64, LaunchResourceGatherInt64Int64)

// Half specialization
#define DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(IndexT, launcher_func)          \
  template <>                                                                \
  void MusaResourceGatherOp<Eigen::half, IndexT>::LaunchKernel(              \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int64_t batch_size, int64_t inner_size, int64_t indices_size,          \
      int64_t params_stride, IndexT limit, musaStream_t stream) {            \
    launcher_func(reinterpret_cast<const void*>(params), indices,            \
                  reinterpret_cast<void*>(output), batch_size, inner_size,   \
                  indices_size, params_stride, limit, stream);               \
  }

DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(int32, LaunchResourceGatherHalfInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(int64, LaunchResourceGatherHalfInt64)

#undef DEFINE_RESOURCE_GATHER_LAUNCHER
#undef DEFINE_RESOURCE_GATHER_LAUNCHER_HALF

// ============================================================================
// ResourceScatterAdd Op (keeps muDNN for atomic operations)
// ============================================================================
template <typename T, typename Index>
struct ResourceScatterAddFastPathLauncher {
  static bool Launch(T*, const Index*, const T*, int64_t, int64_t, Index,
                     musaStream_t) {
    return false;
  }
};

template <>
struct ResourceScatterAddFastPathLauncher<float, int32> {
  static bool Launch(float* params, const int32* indices, const float* updates,
                     int64_t num_updates, int64_t slice_size, int32 limit,
                     musaStream_t stream) {
    LaunchResourceScatterAddRowsFloatInt32(params, indices, updates,
                                           num_updates, slice_size, limit,
                                           stream);
    return true;
  }
};

template <>
struct ResourceScatterAddFastPathLauncher<float, int64> {
  static bool Launch(float* params, const int64* indices, const float* updates,
                     int64_t num_updates, int64_t slice_size, int64 limit,
                     musaStream_t stream) {
    LaunchResourceScatterAddRowsFloatInt64(params, indices, updates,
                                           num_updates, slice_size, limit,
                                           stream);
    return true;
  }
};

template <typename T, typename Index>
class MusaResourceScatterAddOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    if (indices.NumElements() > 0) {
      if (CanUseSparseRowsFastPath(*params, indices, updates)) {
        const int64_t num_updates = indices.NumElements();
        const int64_t slice_size = updates.NumElements() / num_updates;
        const Index limit = static_cast<Index>(params->dim_size(0));
        const bool launched =
            ResourceScatterAddFastPathLauncher<T, Index>::Launch(
                params->flat<T>().data(), indices.flat<Index>().data(),
                updates.flat<T>().data(), num_updates, slice_size, limit,
                GetMusaStreamByCtx(c));
        if (launched) {
          const musaError_t launch_err = musaGetLastError();
          OP_REQUIRES(c, launch_err == musaSuccess,
                      errors::Internal("ResourceScatterAdd fast path launch "
                                       "failed: ",
                                       musaGetErrorString(launch_err)));
          return;
        }
      }

      if (QueryMusaKernelRuntimeView(c).mudnn_handle == nullptr) {
        c->CtxFailure(MusaMudnnHandleRequiredError());
        return;
      }
      auto& h = GetHandleByCtx(c);
      ::musa::dnn::MemoryMaintainer maintainer(
          [](size_t) { return ::musa::dnn::MemoryHandler(); });

      mScatterND op;
      MTOP_CHECK_OK(op.SetMode(mScatterND::Mode::ADD), "SetModeAdd", c);

      auto params_mt = CreateMTensor(*params, format_);
      auto indices_mt = CreateMTensor(indices, format_);
      // Reshape indices for scatter-nd op.
      // indicates the number of axes being scattered (1 for embedding scatter).
      indices_mt.SetNdInfo({static_cast<int64_t>(indices.NumElements()), 1LL});
      auto updates_mt = CreateMTensor(updates, format_);
      MTOP_CHECK_OK_RUN(
          op.Run(h, params_mt, indices_mt, updates_mt, maintainer),
          "RunScatterND", c);
    }
  }

 private:
  bool CanUseSparseRowsFastPath(const Tensor& params, const Tensor& indices,
                                const Tensor& updates) const {
    if (params.dims() < 1 || !TensorShapeUtils::IsVector(indices.shape())) {
      return false;
    }
    if (updates.dims() != params.dims() ||
        updates.dim_size(0) != indices.NumElements()) {
      return false;
    }
    for (int i = 1; i < params.dims(); ++i) {
      if (updates.dim_size(i) != params.dim_size(i)) {
        return false;
      }
    }
    return true;
  }
};

// ============================================================================
// AssignUpdateVariable Op
// ============================================================================

template <typename T, mBinary::Mode BMODE>
class MusaAssignUpdateVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return true; }
  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &variable));
    mutex_lock ml(*variable->mu());

    Tensor* var_tensor = variable->tensor();
    const Tensor& value = c->input(1);

    if (var_tensor->NumElements() > 0) {
      if (QueryMusaKernelRuntimeView(c).mudnn_handle == nullptr) {
        c->CtxFailure(MusaMudnnHandleRequiredError());
        return;
      }
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

// ============================================================================
// VariableShape Op
// ============================================================================

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

// ============================================================================
// Kernel Registration
// ============================================================================

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
REGISTER_MUSA_KERNELS(double);
REGISTER_MUSA_KERNELS(int32);
REGISTER_MUSA_KERNELS(int64);

#define REGISTER_MUSA_ASSIGN_UPDATE_VARIABLE_KERNELS(type)   \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("AssignSubVariableOp")                            \
          .Device(DEVICE_MTGPU)                              \
          .HostMemory("resource")                            \
          .TypeConstraint<type>("dtype"),                    \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::SUB>); \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("AssignAddVariableOp")                            \
          .Device(DEVICE_MTGPU)                              \
          .HostMemory("resource")                            \
          .TypeConstraint<type>("dtype"),                    \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::ADD>);

REGISTER_MUSA_ASSIGN_UPDATE_VARIABLE_KERNELS(Eigen::bfloat16);

REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        MusaVariableShapeOp);

#undef REGISTER_MUSA_KERNELS
#undef REGISTER_MUSA_ASSIGN_UPDATE_VARIABLE_KERNELS

}  // namespace musa
}  // namespace tensorflow
