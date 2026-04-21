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

#include "utils_op.h"

#include "mu/kernel_register.h"

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

static inline mStatus FromMusaError(musaError_t err) {
  if (err == musaSuccess) return mStatus::SUCCESS;
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
  const int rank = static_cast<int>(dims_raw.size());
  // Reuse TensorFlow's shape storage directly instead of copying dims into a
  // temporary vector. For small elementwise ops this shaves a bit of host-side
  // wrapper overhead.
  const int64_t* dims = reinterpret_cast<const int64_t*>(dims_raw.data());

  if (rank >= 4) {
    rst.SetFormat(format);
  } else {
    rst.SetFormat(mFormat::NCHW);
  }

  rst.SetNdInfo(rank, dims);
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

}  // namespace musa
}  // namespace tensorflow
