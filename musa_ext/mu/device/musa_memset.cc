#include "musa_memset.h"

#include "kernel_register.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

mStatus SetTensorInfo(mTensor& tensor, void* device_dst, uint64_t size,
                      int& type_size) {
  type_size = 0;
  MTOP_CHECK_MTDNN_STATUS_RET(tensor.SetAddr(device_dst));

  mTensor::Type tensor_type = mTensor::Type::UINT8;
  uint64_t tensor_size = size;
  if (size & 0x07 == 0) {
    tensor_type = mTensor::Type::UINT64;
    tensor_size = size >> 3;
    type_size = 8;
  } else if (size & 0x03 == 0) {
    tensor_type = mTensor::Type::UINT32;
    tensor_size = size >> 2;
    type_size = 4;
  } else {
    tensor_type = mTensor::Type::UINT8;
    tensor_size = size;
    type_size = 1;
  }
  MTOP_CHECK_MTDNN_STATUS_RET(tensor.SetType(tensor_type));
  if (tensor_size > INT32_MAX) {
    return mStatus::INVALID_PARAMETER;
  }
  MTOP_CHECK_MTDNN_STATUS_RET(
      tensor.SetNdInfo({static_cast<int>(tensor_size)}));

  return mStatus::SUCCESS;
}

mStatus Memset(mHandle& h, void* device_dst, uint64_t size, uint8_t pattern) {
  mTensor tensor;
  int type_size;
  SetTensorInfo(tensor, device_dst, size, type_size);

  ::musa::dnn::Fill op;
  union {
    uint8_t u8_arr[8];
    int64_t i64;
  } fill_op_pattern = {0};
  for (int i = 0; i < 8; ++i) {
    fill_op_pattern.u8_arr[i] = pattern;
  }
  MTOP_CHECK_MTDNN_STATUS_RET(op.SetValue(fill_op_pattern.i64));
  MTOP_CHECK_MTDNN_STATUS_RET(op.Run(h, tensor));

  return mStatus::SUCCESS;
}

mStatus Memset32(mHandle& h, void* device_dst, uint64_t size,
                 uint32_t pattern) {
  mTensor tensor;
  int type_size;
  SetTensorInfo(tensor, device_dst, size, type_size);

  ::musa::dnn::Fill op;
  int64_t fill_op_pattern = 0;
  if (4 == type_size) {
    fill_op_pattern = pattern;
  } else if (8 == type_size) {
    fill_op_pattern = pattern;
    fill_op_pattern = fill_op_pattern << 32 | pattern;
  } else {
    return mStatus::INTERNAL_ERROR;
  }
  MTOP_CHECK_MTDNN_STATUS_RET(op.SetValue(fill_op_pattern));
  MTOP_CHECK_MTDNN_STATUS_RET(op.Run(h, tensor));

  return mStatus::SUCCESS;
}

}  // namespace musa
}  // namespace tensorflow
