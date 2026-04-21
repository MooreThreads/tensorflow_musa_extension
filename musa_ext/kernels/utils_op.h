/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
#define MUSA_PLUGIN_SRC_KERNELS_UTILS_H_

#include <mublas.h>
#include <mudnn.h>
#include <musa_runtime.h>

#include <functional>
#include <utility>
#include <vector>

#include "mu/device/musa_resource_mgr.h"
// These three headers were previously pulled in transitively through
// `mu/device/musa_device.h`, which is now gone. Kernel `.cc` files rely on
// them (shape inference functions, Device* in resource variables, etc.) so we
// include them explicitly here to avoid a cascade of per-file edits.
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

#define DEVICE_MTGPU "MUSA"

// Unified error-handling macros. These predate the PluggableDevice migration
// and are kept verbatim so the 100+ kernels that use them continue to build.
#define MTOP_CHECK_MTDNN_STATUS_RET(status)         \
  do {                                              \
    if ((status) != ::musa::dnn::Status::SUCCESS) { \
      return static_cast<mStatus>(1);               \
    }                                               \
  } while (0)

#define MTOP_CHECK_OK(status, op_name, ctx)                                    \
  do {                                                                         \
    if ((status) != ::musa::dnn::Status::SUCCESS) {                            \
      (ctx)->CtxFailure(errors::Internal(                                      \
          "MUSA ", (op_name), " failed. Status: ", static_cast<int>(status))); \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define MTOP_CHECK_OK_RUN(status, op_name, ctx)                              \
  do {                                                                       \
    auto _status = (status);                                                 \
    if (_status != ::musa::dnn::Status::SUCCESS) {                           \
      (ctx)->CtxFailure(                                                     \
          errors::Internal("MUSA ", (op_name),                               \
                           " failed. Status: ", static_cast<int>(_status))); \
      return;                                                                \
    }                                                                        \
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

// Wraps a `size -> MemoryHandler` lambda as a `MemoryMaintainer`, the type
// expected by mudnn ops for scratch-space allocation. Kernels used to obtain
// this from `MusaDevice::GetMemMaintainer`, which was just a typed
// passthrough — this helper preserves that call site ergonomics while the
// MusaDevice class itself is gone.
inline ::musa::dnn::MemoryMaintainer MakeMusaMemMaintainer(
    std::function<::musa::dnn::MemoryHandler(size_t)> fn) {
  return ::musa::dnn::MemoryMaintainer(std::move(fn));
}

class MusaOpKernel : public OpKernel {
 public:
  explicit MusaOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    format_ = GetMusaFormat(ctx);
  }

 protected:
  mFormat format_;
};

// Thread-local cache for current device to avoid redundant musaSetDevice
// calls on the fast path. Kernels call this before issuing any MUSA API so
// that the driver thread-local current device matches the one TF asked us to
// run on.
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

// Retrieves the MUSA driver stream backing the OpKernelContext.
//
// Under the PluggableDevice C API, TF core wraps our SP_Stream in a
// `CStream` object whose `StreamInterface::GpuStreamHack()` returns the
// plugin's opaque `SP_Stream` value verbatim. Since we stash the raw
// `musaStream_t` in `SP_Stream` (see musa_se_callbacks.cc), reinterpret_cast
// recovers the driver-level handle with no indirection — the same trick TF's
// own GPU code uses in tensorflow/stream_executor/gpu/gpu_stream.h.
inline musaStream_t GetMusaStreamByCtx(tensorflow::OpKernelContext* context) {
  if (context == nullptr) return nullptr;
  auto* dev_ctx = context->op_device_context();
  if (!dev_ctx || !dev_ctx->stream()) return nullptr;
  return reinterpret_cast<musaStream_t>(
      dev_ctx->stream()->implementation()->GpuStreamHack());
}

// Returns the device ordinal that is currently dispatching this kernel.
//
// TF's StreamExecutor tracks the device ordinal on the StreamExecutor
// backing each Stream, so walking Stream -> parent -> device_ordinal() is
// the canonical way to recover it without any plugin-specific cast.
inline int GetMusaDeviceIdByCtx(tensorflow::OpKernelContext* context) {
  if (context == nullptr) return -1;
  auto* dev_ctx = context->op_device_context();
  if (!dev_ctx || !dev_ctx->stream()) return -1;
  return dev_ctx->stream()->parent()->device_ordinal();
}

// Returns a mudnn handle bound to this kernel's stream.
//
// MusaResourceMgr keeps one mudnn::Handle per device ordinal and re-binds
// the stream on every call. This matches TF's cuDNN wrapper in
// tensorflow/stream_executor/cuda/cuda_dnn.cc, which also rebinds on every
// kernel launch — a small price for having a single handle per device.
inline ::musa::dnn::Handle& GetHandleByCtx(
    tensorflow::OpKernelContext* context) {
  int device_id = GetMusaDeviceIdByCtx(context);
  CachedMusaSetDevice(device_id);
  musaStream_t stream = GetMusaStreamByCtx(context);
  return MusaResourceMgr::Instance().GetMudnnHandle(device_id, stream);
}

// Returns an mublas handle bound to this kernel's stream. See the note on
// GetHandleByCtx — the handle is process-global per device, not per stream.
inline mublasHandle_t GetMublasHandleByCtx(
    tensorflow::OpKernelContext* context) {
  int device_id = GetMusaDeviceIdByCtx(context);
  CachedMusaSetDevice(device_id);
  musaStream_t stream = GetMusaStreamByCtx(context);
  return MusaResourceMgr::Instance().GetMublasHandle(device_id, stream);
}

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
