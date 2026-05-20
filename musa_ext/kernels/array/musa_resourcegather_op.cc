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

#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

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
void LaunchResourceGatherBFloat16Int32(
    const void* params, const int* indices, void* output, int64_t batch_size,
    int64_t inner_size, int64_t indices_size, int64_t params_stride, int limit,
    musaStream_t stream);
void LaunchResourceGatherBFloat16Int64(
    const void* params, const int64_t* indices, void* output,
    int64_t batch_size, int64_t inner_size, int64_t indices_size,
    int64_t params_stride, int64_t limit, musaStream_t stream);
void LaunchResourceGatherFloatToBFloat16Int32(
    const float* params, const int* indices, void* output,
    int64_t batch_size, int64_t inner_size, int64_t indices_size,
    int64_t params_stride, int limit, musaStream_t stream);
void LaunchResourceGatherFloatToBFloat16Int64(
    const float* params, const int64_t* indices, void* output,
    int64_t batch_size, int64_t inner_size, int64_t indices_size,
    int64_t params_stride, int64_t limit, musaStream_t stream);
void LaunchCriteoSparseEmbeddingGatherFloatInt32(
    const uintptr_t* params_ptrs, const int* limits, const int* indices,
    float* output, int batch_size, int feature_count, int inner_size,
    musaStream_t stream);
void LaunchResourceScatterSubBFloat16Int32(
    float* params, const int* indices, const void* updates, float alpha,
    int64_t indices_size, int64_t inner_size, int64_t limit,
    musaStream_t stream);
void LaunchResourceScatterSubBFloat16Int64(
    float* params, const int64_t* indices, const void* updates, float alpha,
    int64_t indices_size, int64_t inner_size, int64_t limit,
    musaStream_t stream);
}

namespace tensorflow {
namespace musa {

Status ReadFloatScalarHostOrDevice(const Tensor& tensor, float* value) {
  const float* data = tensor.flat<float>().data();

  musaPointerAttributes attributes;
  musaError_t attr_err = musaPointerGetAttributes(&attributes, data);
  const bool is_device =
      attr_err == musaSuccess && attributes.type == musaMemoryTypeDevice;
  if (attr_err != musaSuccess) {
    musaGetLastError();
  }

  if (is_device) {
    musaError_t copy_err =
        musaMemcpy(value, data, sizeof(float), musaMemcpyDeviceToHost);
    if (copy_err != musaSuccess) {
      return errors::Internal("Failed to copy float scalar to host");
    }
    return OkStatus();
  }

  *value = data[0];
  return OkStatus();
}

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

// BFloat16 specialization uses the same 16-bit gather ABI as half.
template <>
void MusaResourceGatherOp<Eigen::bfloat16, int32>::LaunchKernel(
    const Eigen::bfloat16* params, const int32* indices,
    Eigen::bfloat16* output, int64_t batch_size, int64_t inner_size,
    int64_t indices_size, int64_t params_stride, int32 limit,
    musaStream_t stream) {
  LaunchResourceGatherBFloat16Int32(
      reinterpret_cast<const void*>(params), indices,
      reinterpret_cast<void*>(output), batch_size, inner_size, indices_size,
      params_stride, limit, stream);
}

template <>
void MusaResourceGatherOp<Eigen::bfloat16, int64>::LaunchKernel(
    const Eigen::bfloat16* params, const int64* indices,
    Eigen::bfloat16* output, int64_t batch_size, int64_t inner_size,
    int64_t indices_size, int64_t params_stride, int64 limit,
    musaStream_t stream) {
  LaunchResourceGatherBFloat16Int64(
      reinterpret_cast<const void*>(params), indices,
      reinterpret_cast<void*>(output), batch_size, inner_size, indices_size,
      params_stride, limit, stream);
}

#undef DEFINE_RESOURCE_GATHER_LAUNCHER
#undef DEFINE_RESOURCE_GATHER_LAUNCHER_HALF

REGISTER_OP("MusaResourceGatherFloatToBFloat16")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Output("output: bfloat16")
    .Attr("Tindices: {int32, int64}")
    .Attr("batch_dims: int = 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return OkStatus();
    });

template <typename Index>
class MusaResourceGatherFloatToBFloat16Op : public MusaOpKernel {
 public:
  explicit MusaResourceGatherFloatToBFloat16Op(OpKernelConstruction* c)
      : MusaOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));

    tf_shared_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);

    OP_REQUIRES(
        c, params.dtype() == DT_FLOAT,
        errors::InvalidArgument(
            "MusaResourceGatherFloatToBFloat16 expects float resource"));
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
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &out));
    if (out->NumElements() == 0 || indices.NumElements() == 0) return;

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

    LaunchKernel(params.flat<float>().data(), indices.flat<Index>().data(),
                 reinterpret_cast<void*>(out->flat<bfloat16>().data()),
                 batch_size, inner_size, indices_size, params_stride, limit,
                 GetMusaStreamByCtx(c));
  }

 private:
  int32 batch_dims_ = 0;

  void LaunchKernel(const float* params, const Index* indices, void* output,
                    int64_t batch_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, Index limit,
                    musaStream_t stream);
};

REGISTER_OP("MusaCriteoSparseEmbeddingGather")
    .Input("resources: N * resource")
    .Input("indices: int32")
    .Output("output: float")
    .Attr("N: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return OkStatus();
    });

class MusaCriteoSparseEmbeddingGatherOp : public MusaOpKernel {
 public:
  explicit MusaCriteoSparseEmbeddingGatherOp(OpKernelConstruction* c)
      : MusaOpKernel(c), cache_initialized_(false) {
    OP_REQUIRES_OK(c, c->GetAttr("N", &feature_count_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    OP_REQUIRES(c, feature_count_ > 0,
                errors::InvalidArgument("N must be positive"));
    OP_REQUIRES(c, c->num_inputs() == feature_count_ + 1,
                errors::InvalidArgument(
                    "MusaCriteoSparseEmbeddingGather expects N resources "
                    "plus one indices input, got ",
                    c->num_inputs()));

    const Tensor& indices = c->input(feature_count_);
    OP_REQUIRES(c, indices.dims() == 2,
                errors::InvalidArgument(
                    "indices must be rank 2 [batch, features], got ",
                    indices.shape().DebugString()));
    OP_REQUIRES(c, indices.dim_size(1) == feature_count_,
                errors::InvalidArgument("indices feature dimension mismatch: ",
                                        indices.dim_size(1), " vs ",
                                        feature_count_));

    const int batch_size = static_cast<int>(indices.dim_size(0));
    int inner_size = -1;
    using SharedLock = tf_shared_lock;
    std::vector<core::RefCountPtr<Var>> vars(feature_count_);
    std::vector<std::unique_ptr<SharedLock>> locks;
    locks.reserve(feature_count_);
    std::vector<uintptr_t> host_ptrs(feature_count_);
    std::vector<int> host_limits(feature_count_);

    for (int i = 0; i < feature_count_; ++i) {
      OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, i), &vars[i]));
      locks.emplace_back(new SharedLock(*vars[i]->mu()));
      const Tensor& params = *vars[i]->tensor();
      OP_REQUIRES(c, params.dtype() == DT_FLOAT,
                  errors::InvalidArgument(
                      "MusaCriteoSparseEmbeddingGather expects float resource"));
      OP_REQUIRES(c, params.dims() == 2,
                  errors::InvalidArgument(
                      "embedding resource must be rank 2, got ",
                      params.shape().DebugString()));
      if (inner_size < 0) {
        inner_size = static_cast<int>(params.dim_size(1));
      }
      OP_REQUIRES(c, params.dim_size(1) == inner_size,
                  errors::InvalidArgument(
                      "embedding dimensions must match, got ",
                      params.dim_size(1), " vs ", inner_size));
      OP_REQUIRES(c, params.dim_size(0) <= std::numeric_limits<int>::max(),
                  errors::InvalidArgument(
                      "embedding row count exceeds int limit: ",
                      params.dim_size(0)));
      host_ptrs[i] =
          reinterpret_cast<uintptr_t>(params.flat<float>().data());
      host_limits[i] = static_cast<int>(params.dim_size(0));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(
                          0, TensorShape({batch_size, feature_count_,
                                          inner_size}),
                          &output));
    if (output->NumElements() == 0) return;

    musaStream_t stream = GetMusaStreamByCtx(c);
    musaError_t err = musaSuccess;
    {
      mutex_lock lock(mu_);
      const bool cache_changed =
          !cache_initialized_ || cached_ptrs_ != host_ptrs ||
          cached_limits_ != host_limits;
      if (cache_changed) {
        if (!cache_initialized_) {
          AllocatorAttributes attr;
          attr.set_on_host(false);
          OP_REQUIRES_OK(c, c->allocate_temp(DT_UINT64,
                                             TensorShape({feature_count_}),
                                             &device_ptrs_, attr));
          OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32,
                                             TensorShape({feature_count_}),
                                             &device_limits_, attr));
        }

        err = musaMemcpyAsync(device_ptrs_.flat<uint64>().data(),
                              host_ptrs.data(),
                              host_ptrs.size() * sizeof(uintptr_t),
                              musaMemcpyHostToDevice, stream);
        OP_REQUIRES(c, err == musaSuccess,
                    errors::Internal("copy embedding pointer table failed: ",
                                     musaGetErrorString(err)));
        err = musaMemcpyAsync(device_limits_.flat<int32>().data(),
                              host_limits.data(),
                              host_limits.size() * sizeof(int),
                              musaMemcpyHostToDevice, stream);
        OP_REQUIRES(c, err == musaSuccess,
                    errors::Internal("copy embedding limit table failed: ",
                                     musaGetErrorString(err)));
        err = musaStreamSynchronize(stream);
        OP_REQUIRES(c, err == musaSuccess,
                    errors::Internal("sync embedding table metadata failed: ",
                                     musaGetErrorString(err)));
        cached_ptrs_ = host_ptrs;
        cached_limits_ = host_limits;
        cache_initialized_ = true;
      }
    }

    LaunchCriteoSparseEmbeddingGatherFloatInt32(
        reinterpret_cast<const uintptr_t*>(device_ptrs_.flat<uint64>().data()),
        device_limits_.flat<int32>().data(), indices.flat<int32>().data(),
        output->flat<float>().data(), batch_size, feature_count_, inner_size,
        stream);
    err = musaGetLastError();
    OP_REQUIRES(c, err == musaSuccess,
                errors::Internal(
                    "MusaCriteoSparseEmbeddingGather launch failed: ",
                    musaGetErrorString(err)));
  }

 private:
  int feature_count_ = 0;
  bool cache_initialized_;
  Tensor device_ptrs_;
  Tensor device_limits_;
  std::vector<uintptr_t> cached_ptrs_;
  std::vector<int> cached_limits_;
  mutex mu_;
};

template <>
void MusaResourceGatherFloatToBFloat16Op<int32>::LaunchKernel(
    const float* params, const int32* indices, void* output,
    int64_t batch_size, int64_t inner_size, int64_t indices_size,
    int64_t params_stride, int32 limit, musaStream_t stream) {
  LaunchResourceGatherFloatToBFloat16Int32(
      params, indices, output, batch_size, inner_size, indices_size,
      params_stride, limit, stream);
}

template <>
void MusaResourceGatherFloatToBFloat16Op<int64>::LaunchKernel(
    const float* params, const int64* indices, void* output,
    int64_t batch_size, int64_t inner_size, int64_t indices_size,
    int64_t params_stride, int64 limit, musaStream_t stream) {
  LaunchResourceGatherFloatToBFloat16Int64(
      params, reinterpret_cast<const int64_t*>(indices), output, batch_size,
      inner_size, indices_size, params_stride, limit, stream);
}

// ============================================================================
// ResourceScatterAdd Op (keeps muDNN for atomic operations)
// ============================================================================
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
};

REGISTER_OP("MusaResourceScatterSubBFloat16")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: Tgrad")
    .Input("alpha: T")
    .Attr("T: {float}")
    .Attr("Tgrad: {bfloat16}")
    .Attr("Tindices: {int32, int64}")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

template <typename Index>
class MusaResourceScatterSubBFloat16Op : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    OP_REQUIRES(c, v->tensor()->IsInitialized(),
                errors::FailedPrecondition("Variable not initialized."));

    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    const Tensor& alpha = c->input(3);

    OP_REQUIRES(c, params->dtype() == DT_FLOAT,
                errors::InvalidArgument(
                    "MusaResourceScatterSubBFloat16 expects float resource"));
    OP_REQUIRES(c, updates.dtype() == DT_BFLOAT16,
                errors::InvalidArgument(
                    "MusaResourceScatterSubBFloat16 expects bfloat16 updates"));
    OP_REQUIRES(c, alpha.NumElements() == 1,
                errors::InvalidArgument("alpha must be a scalar, got ",
                                        alpha.NumElements(), " elements"));
    OP_REQUIRES(c, params->dims() >= 1,
                errors::InvalidArgument("resource must be at least 1D"));

    const int64_t indices_size = indices.NumElements();
    if (indices_size == 0 || updates.NumElements() == 0) return;

    const int64_t limit = params->dim_size(0);
    const int64_t inner_size = params->NumElements() / limit;
    OP_REQUIRES(c, updates.NumElements() == indices_size * inner_size,
                errors::InvalidArgument(
                    "updates shape does not match indices/resource. updates: ",
                    updates.shape().DebugString(),
                    " indices: ", indices.shape().DebugString(),
                    " resource: ", params->shape().DebugString()));

    float alpha_value = 0.0f;
    OP_REQUIRES_OK(c, ReadFloatScalarHostOrDevice(alpha, &alpha_value));

    musaStream_t stream = GetMusaStreamByCtx(c);
    LaunchKernel(params->flat<float>().data(), indices.flat<Index>().data(),
                 reinterpret_cast<const void*>(updates.flat<bfloat16>().data()),
                 alpha_value, indices_size, inner_size, limit, stream);
  }

 private:
  void LaunchKernel(float* params, const Index* indices, const void* updates,
                    float alpha, int64_t indices_size, int64_t inner_size,
                    int64_t limit, musaStream_t stream);
};

template <>
void MusaResourceScatterSubBFloat16Op<int32>::LaunchKernel(
    float* params, const int32* indices, const void* updates, float alpha,
    int64_t indices_size, int64_t inner_size, int64_t limit,
    musaStream_t stream) {
  LaunchResourceScatterSubBFloat16Int32(
      params, indices, updates, alpha, indices_size, inner_size, limit, stream);
}

template <>
void MusaResourceScatterSubBFloat16Op<int64>::LaunchKernel(
    float* params, const int64* indices, const void* updates, float alpha,
    int64_t indices_size, int64_t inner_size, int64_t limit,
    musaStream_t stream) {
  LaunchResourceScatterSubBFloat16Int64(
      params, reinterpret_cast<const int64_t*>(indices), updates, alpha,
      indices_size, inner_size, limit, stream);
}

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
REGISTER_MUSA_KERNELS(Eigen::bfloat16);
REGISTER_MUSA_KERNELS(double);
REGISTER_MUSA_KERNELS(int32);
REGISTER_MUSA_KERNELS(int64);

#define REGISTER_MUSA_GATHER_FLOAT_TO_BFLOAT16(type)                   \
  REGISTER_KERNEL_BUILDER(Name("MusaResourceGatherFloatToBFloat16")    \
                              .Device(DEVICE_MTGPU)                   \
                              .HostMemory("resource")                 \
                              .TypeConstraint<type>("Tindices"),      \
                          MusaResourceGatherFloatToBFloat16Op<type>);

REGISTER_MUSA_GATHER_FLOAT_TO_BFLOAT16(int32);
REGISTER_MUSA_GATHER_FLOAT_TO_BFLOAT16(int64);

REGISTER_KERNEL_BUILDER(Name("MusaCriteoSparseEmbeddingGather")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("resources"),
                        MusaCriteoSparseEmbeddingGatherOp);

#define REGISTER_MUSA_SCATTER_SUB_BFLOAT16(type)                     \
  REGISTER_KERNEL_BUILDER(Name("MusaResourceScatterSubBFloat16")      \
                              .Device(DEVICE_MTGPU)                  \
                              .HostMemory("resource")                \
                              .HostMemory("alpha")                   \
                              .TypeConstraint<float>("T")            \
                              .TypeConstraint<bfloat16>("Tgrad")     \
                              .TypeConstraint<type>("Tindices"),     \
                          MusaResourceScatterSubBFloat16Op<type>);

REGISTER_MUSA_SCATTER_SUB_BFLOAT16(int32);
REGISTER_MUSA_SCATTER_SUB_BFLOAT16(int64);

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

REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        MusaVariableShapeOp);

#undef REGISTER_MUSA_KERNELS
#undef REGISTER_MUSA_GATHER_FLOAT_TO_BFLOAT16
#undef REGISTER_MUSA_SCATTER_SUB_BFLOAT16
#undef REGISTER_MUSA_ASSIGN_UPDATE_VARIABLE_KERNELS

}  // namespace musa
}  // namespace tensorflow
