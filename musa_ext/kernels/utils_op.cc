#include "utils_op.h"

#include "mu/kernel_register.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
namespace musa {

namespace {
mType GetType(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_UINT8:
      return mType::UINT8;
    case DataType::DT_INT16:
      return mType::INT16;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_UINT16:
      return mType::UINT16;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_UINT32:
      return mType::UINT32;
    case DataType::DT_UINT64:
      return mType::UINT64;
    case DataType::DT_BOOL:
      return mType::BOOL;
    default:
      CHECK(false);
      throw;
  }
}
}  // namespace

// Helper function to convert musaError_t to mStatus (mudnn Status)
static inline mStatus FromMusaError(musaError_t err) {
  if (err == musaSuccess) return mStatus::SUCCESS;
  // mudnn Status doesn't have OUT_OF_MEMORY, use INTERNAL_ERROR for all errors
  return mStatus::INTERNAL_ERROR;
}

mStatus MusaFree(void* ptr) {
  if (ptr) {
    musaError_t err = musaFree(ptr);
    return FromMusaError(err);
  }
  return mStatus::SUCCESS;
}

mStatus MusaAllocate(size_t size, void** ptr) {
  musaError_t err = musaMalloc(ptr, size);
  return FromMusaError(err);
}

mTensor CreateMTensor(const Tensor& t, mFormat format) {
  mTensor rst;
  rst.SetAddr(
      const_cast<void*>(static_cast<const void*>(t.tensor_data().data())));
  rst.SetType(GetType(t.dtype()));

  auto dims_raw = t.shape().dim_sizes();
  std::vector<int64_t> dims;
  for (auto d : dims_raw) {
    dims.push_back(d);
  }

  if (dims.size() >= 4) {
    rst.SetFormat(format);
  } else {
    rst.SetFormat(mFormat::NCHW);
  }

  rst.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  return rst;
}

mTensor CreateMTensor(const Tensor& t) {
  mTensor rst;
  CHECK(rst.SetAddr(t.data()) == ::musa::dnn::Status::SUCCESS)
      << "SetAddr failed";
  CHECK(rst.SetType(GetType(t.dtype())) == ::musa::dnn::Status::SUCCESS)
      << "SetType failed";
  auto dims_int = t.shape().dim_sizes();
  CHECK(rst.SetNdInfo(static_cast<int>(dims_int.size()),
                      reinterpret_cast<const int64_t*>(dims_int.data())) ==
        ::musa::dnn::Status::SUCCESS)
      << "SetNdInfo failed";
  return rst;
}

mFormat GetMusaFormat(OpKernelConstruction* ctx) {
  string df;
  if (ctx->HasAttr("data_format")) {
    if (ctx->GetAttr("data_format", &df).ok()) {
      return (df == "NCHW") ? mFormat::NCHW : mFormat::NHWC;
    }
  }
  return mFormat::NHWC;
}

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context) {
  DeviceBase* device_base = context->device();
  if (!device_base) {
    LOG(ERROR) << "GetDeviceByCtx: device_base is null";
    return nullptr;
  }
  MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(device_base);
  if (!musa_device) {
    LOG(ERROR) << "GetDeviceByCtx: musa_device is null";
    return nullptr;
  }
  // Note: musaSetDevice is called in GetHandleByCtx with caching
  // We skip it here to avoid redundant calls
  return musa_device;
}

}  // namespace musa
}  // namespace tensorflow
