#ifndef MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
#define MUSA_PLUGIN_SRC_KERNELS_UTILS_H_

#include <mudnn.h>

#include <vector>

#include "kernels/musa_kernel_runtime.h"
#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#define DEVICE_MTGPU "MUSA"

namespace tensorflow {
namespace musa {
// Legacy: kernels that still require a concrete C++ `MusaDevice` (not just MuDNN).
inline Status MusaCppDevicePathRequiredError() {
  return errors::Unimplemented(
      "MUSA op requires the C++ MusaDevice path (import tensorflow_musa or "
      "tf.load_op_library with MUSA_ENABLE_SE_PLUGIN unset). "
      "SE-only PluggableDevice (MUSA_ENABLE_SE_PLUGIN=1) is not supported for "
      "this kernel; use the default C++ device path.");
}

// Prefer this for MuDNN-backed kernels: permits either legacy `MusaDevice` or SE
// registry–initialized `MusaKernelRuntimeView::mudnn_handle`.
inline Status MusaMudnnHandleRequiredError() {
  return errors::Unimplemented(
      "MUSA MuDNN kernel requires `MusaKernelRuntimeView::mudnn_handle`: use "
      "the default C++ MusaDevice path, or ensure the PluggableDevice SE "
      "runtime has registered the device ordinal and streams are available.");
}
}  // namespace musa
}  // namespace tensorflow

#define MUSA_OP_REQUIRES_MUDNN_HANDLE(ctx)                                      \
  OP_REQUIRES(                                                                   \
      (ctx),                                                                     \
      ::tensorflow::musa::QueryMusaKernelRuntimeView((ctx)).mudnn_handle !=     \
          nullptr,                                                               \
      ::tensorflow::musa::MusaMudnnHandleRequiredError())

// Deprecated for MuDNN ops: prefer MUSA_OP_REQUIRES_MUDNN_HANDLE.
#define MUSA_OP_REQUIRES_CPP_MUSA_DEVICE(ctx)                \
  OP_REQUIRES((ctx),                                         \
              ::tensorflow::musa::TryGetMusaDeviceFromContext( \
                  (ctx)) != nullptr,                         \
              ::tensorflow::musa::MusaCppDevicePathRequiredError())

// 统一的错误处理宏
#define MTOP_CHECK_MTDNN_STATUS_RET(status)         \
  do {                                              \
    if ((status) != ::musa::dnn::Status::SUCCESS) { \
      return static_cast<mStatus>(1);               \
    }                                               \
  } while (0)

#define MTOP_CHECK_OK(mudnn_call_status, op_name, mctx)                      \
  do {                                                                       \
    if ((mctx) != nullptr && !((mctx)->status().ok())) {                    \
      return;                                                                \
    }                                                                         \
    if ((mudnn_call_status) != ::musa::dnn::Status::SUCCESS) {              \
      (mctx)->CtxFailure(                                                   \
          errors::Internal("MUSA ", (op_name), " failed. Status: ",        \
                            static_cast<int>(mudnn_call_status)));         \
      return;                                                                \
    }                                                                         \
  } while (0)

#define MTOP_CHECK_OK_RUN(mudnn_call_status, op_name, mctx)                 \
  do {                                                                      \
    if ((mctx) != nullptr && !((mctx)->status().ok())) {                   \
      return;                                                               \
    }                                                                       \
    auto _mudnn_st = (mudnn_call_status);                                 \
    if (_mudnn_st != ::musa::dnn::Status::SUCCESS) {                     \
      (mctx)->CtxFailure(                                                \
          errors::Internal("MUSA ", (op_name),                          \
                           " failed. Status: ", static_cast<int>(_mudnn_st))); \
      return;                                                            \
    }                                                                       \
  } while (0)

namespace tensorflow {
namespace musa {

using mHandle = ::musa::dnn::Handle;
using mTensor = ::musa::dnn::Tensor;
using mType = ::musa::dnn::Tensor::Type;
using mFormat = ::musa::dnn::Tensor::Format;
using mStatus = ::musa::dnn::Status;

using mUnary = ::musa::dnn::Unary;
using UNARY_MODE = ::musa::dnn::Unary::Mode;
using mBinary = ::musa::dnn::Binary;
using BINARY_MODE = ::musa::dnn::Binary::Mode;
using mTernary = ::musa::dnn::Ternary;
using mFill = ::musa::dnn::Fill;
using mReduce = ::musa::dnn::Reduce;
using mConcat = ::musa::dnn::Concat;
using mPad = ::musa::dnn::Pad;
using mPermute = ::musa::dnn::Permute;

using mConvolution = ::musa::dnn::Convolution;
using mPooling = ::musa::dnn::Pooling;
using mSoftmax = ::musa::dnn::Softmax;
using SOFTMAX_MODE = ::musa::dnn::Softmax::Mode;
using mBatchNorm = ::musa::dnn::BatchNorm;
using mGroupNorm = ::musa::dnn::GroupNorm;
using mLayerNorm = ::musa::dnn::LayerNorm;
using mDropout = ::musa::dnn::Dropout;

using mMatMul = ::musa::dnn::MatMul;
using mBatchMatMul = ::musa::dnn::BatchMatMul;

using mGatherX = ::musa::dnn::GatherX;
using mScatter = ::musa::dnn::Scatter;
using mScatterND = ::musa::dnn::ScatterND;
using mCum = ::musa::dnn::Cum;
using mTopK = ::musa::dnn::TopK;
using mUnique = ::musa::dnn::Unique;

mTensor CreateMTensor(const Tensor& t, mFormat format);
mTensor CreateMTensor(const Tensor& t);

mStatus MusaFree(void* ptr);
mStatus MusaAllocate(size_t size, void** ptr);

mFormat GetMusaFormat(OpKernelConstruction* ctx);

class MusaOpKernel : public OpKernel {
 public:
  explicit MusaOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    format_ = GetMusaFormat(ctx);
  }

 protected:
  mFormat format_;
};

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context);

inline int GetMusaDeviceIdByCtx(tensorflow::OpKernelContext* context) {
  return GetMusaDeviceIdForKernelContext(context);
}

// Not for real use: only returned with context already in failed state after
// CtxFailure. Prefer MUSA_OP_REQUIRES_CPP_MUSA_DEVICE or MTOP_* after
// GetHandleByCtx, which return early when !ctx->status().ok().
inline ::musa::dnn::Handle& MudnnHandleOrSinkAfterCtxFailure() {
  // Deliberately leaked sink handle for error paths after CtxFailure.
  // This is never used for real kernel execution.
  static ::musa::dnn::Handle* h = new ::musa::dnn::Handle();
  return *h;
}

// Thread-local cache for current device to avoid redundant musaSetDevice calls
// Assumption: per-thread device assignment stays stable for kernel execution.
inline musaError_t CachedMusaSetDevice(int device_id) {
  static thread_local int cached_device_id = -1;
  if (device_id != cached_device_id) {
    musaError_t err = musaSetDevice(device_id);
    if (err == musaSuccess) {
      cached_device_id = device_id;
    }
    return err;
  }
  return musaSuccess;
}

inline ::musa::dnn::Handle& GetHandleByCtx(
    tensorflow::OpKernelContext* context) {
  MusaKernelRuntimeView view = QueryMusaKernelRuntimeView(context);
  if (view.mudnn_handle != nullptr) {
    if (view.device_id >= 0) {
      musaError_t err = CachedMusaSetDevice(view.device_id);
      if (err != musaSuccess) {
        if (context != nullptr) {
          context->CtxFailure(errors::Internal(
              "musaSetDevice(", view.device_id, ") failed: ",
              musaGetErrorString(err)));
        }
        return MudnnHandleOrSinkAfterCtxFailure();
      }
    }
    return *view.mudnn_handle;
  }
  if (context != nullptr) {
    context->CtxFailure(MusaMudnnHandleRequiredError());
  }
  return MudnnHandleOrSinkAfterCtxFailure();
}

inline musaStream_t GetMusaStreamByCtx(tensorflow::OpKernelContext* context) {
  return GetMusaStreamForKernelContext(context);
}

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
