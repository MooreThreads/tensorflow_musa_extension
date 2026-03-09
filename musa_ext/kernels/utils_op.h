#ifndef MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
#define MUSA_PLUGIN_SRC_KERNELS_UTILS_H_

#include <mudnn.h>

#include <vector>

#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#define DEVICE_MTGPU "MUSA"

// 统一的错误处理宏
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

class MusaOpKernel : public OpKernel {
 public:
  explicit MusaOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    format_ = GetMusaFormat(ctx);
  }

 protected:
  mFormat format_;
};

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context);

// Thread-local cache for current device to avoid redundant musaSetDevice calls
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
  auto* musa_device = static_cast<MusaDevice*>(context->device());
  int device_id = musa_device->get_device_id();

  musaError_t err = CachedMusaSetDevice(device_id);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaSetDevice failed: " << musaGetErrorString(err);
  }

  return musa_device->mudnn_handle();
}

inline musaStream_t GetMusaStreamByCtx(tensorflow::OpKernelContext* context) {
  auto* musa_device = static_cast<MusaDevice*>(context->device());
  if (!musa_device) return nullptr;
  return musa_device->GetStream();
}

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_UTILS_H_
