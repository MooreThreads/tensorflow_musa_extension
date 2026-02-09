#include "utils_op.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
namespace musa {

namespace {
mType GetType(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT: return mType::FLOAT;
    case DataType::DT_DOUBLE: return mType::DOUBLE;
    case DataType::DT_INT32: return mType::INT32;
    case DataType::DT_UINT8: return mType::UINT8;
    case DataType::DT_INT16: return mType::INT16;
    case DataType::DT_INT8: return mType::INT8;
    case DataType::DT_INT64: return mType::INT64;
    case DataType::DT_BFLOAT16: return mType::BFLOAT16;
    case DataType::DT_UINT16: return mType::UINT16;
    case DataType::DT_HALF: return mType::HALF;
    case DataType::DT_UINT32: return mType::UINT32;
    case DataType::DT_UINT64: return mType::UINT64;
    case DataType::DT_BOOL: return mType::BOOL;
    default: CHECK(false); throw;
  }
}
}

mStatus MusaFree(void* ptr) { if (ptr) musaFree(ptr); return mStatus::SUCCESS; }
mStatus MusaAllocate(size_t size, void** ptr) { musaMalloc(ptr, size); return mStatus::SUCCESS; }


mTensor CreateMTensor(const Tensor& t, mFormat format) {
  mTensor rst;
  rst.SetAddr(const_cast<void*>(static_cast<const void*>(t.tensor_data().data())));
  rst.SetType(GetType(t.dtype()));

  // 1. 原样提取维度 (不做任何补齐，完全信任底层对低维的支持)
  auto dims_raw = t.shape().dim_sizes();
  std::vector<int64_t> dims;
  for (auto d : dims_raw) {
      dims.push_back(d);
  }

  // 2. 格式策略 (你的核心想法)

  if (dims.size() >= 4) {
      rst.SetFormat(format);
  }

  else {
      rst.SetFormat(mFormat::NCHW);
  }

  rst.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  return rst;
}

// 对于 不是4d的算子
mTensor CreateMTensor(const Tensor& t) {
  
  mTensor rst;
  MTOP_CHECK_LOG(rst.SetAddr(t.data()), "SetAddr");
  MTOP_CHECK_LOG(rst.SetType(GetType(t.dtype())), "SetType");
  auto dims_int = t.shape().dim_sizes();
  MTOP_CHECK_LOG(rst.SetNdInfo(static_cast<int>(dims_int.size()), reinterpret_cast<const int64_t*>(dims_int.data())), "SetNdInfo");
  return rst;
}

// ============================================================
// GetMusaFormat: 标准逻辑
// ============================================================
mFormat GetMusaFormat(OpKernelConstruction* ctx) {
  string df;
  // 正常读取属性，不多做手脚
  if (ctx->HasAttr("data_format")) {
    if (ctx->GetAttr("data_format", &df).ok()) {
      return (df == "NCHW") ? mFormat::NCHW : mFormat::NHWC;
    }
  }
  // 默认 NHWC (照顾 4D 算子)
  return mFormat::NHWC; 
}

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context) {
  DeviceBase* device_base = context->device();
  MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(device_base);
  if (!musa_device) return nullptr;
  musaSetDevice(musa_device->get_device_id());
  return musa_device;
}

}  // namespace musa
}  // namespace tensorflow

