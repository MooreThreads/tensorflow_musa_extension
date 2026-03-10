#include "musa_device.h"

#include <iostream>

#include "mu/device/musa_event.h"
#include "mu/device/musa_stream.h"
#include "musa_allocator.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

MusaDeviceContext::MusaDeviceContext(
    musaStream_t stream, ::stream_executor::StreamExecutor* executor)
    : stream_handle_(stream) {
  implementation_ = new ::stream_executor::musa::MusaStream(stream);
  official_stream_ = new ::stream_executor::Stream(executor, implementation_);
  official_stream_->Init();
}

MusaDeviceContext::~MusaDeviceContext() {
  if (official_stream_) {
    delete official_stream_;
  }
}

void MusaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                              Device* device,
                                              Tensor* device_tensor,
                                              StatusCallback done,
                                              bool sync_dst_compute) const {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  musaSetDevice(musa_dev->get_device_id());

  const void* src = cpu_tensor->tensor_data().data();
  void* dst = const_cast<char*>(device_tensor->tensor_data().data());
  size_t bytes = cpu_tensor->TotalBytes();

  if (bytes > 0) {
    mStatus m_stat = MusaMemcpyAsyncH2D(dst, src, bytes, stream_handle_);
    if (m_stat != mStatus::SUCCESS) {
      done(errors::Internal("MUSA H2D async copy failed."));
      return;
    }
  }
  done(Status::OK());
}

void MusaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              StringPiece tensor_name,
                                              Device* device,
                                              Tensor* cpu_tensor,
                                              StatusCallback done) {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  musaSetDevice(musa_dev->get_device_id());

  const void* src = device_tensor->tensor_data().data();
  void* dst = const_cast<char*>(cpu_tensor->tensor_data().data());
  size_t bytes = device_tensor->TotalBytes();

  if (bytes > cpu_tensor->TotalBytes()) {
    bytes = cpu_tensor->TotalBytes();
  }

  if (bytes > 0) {
    mStatus m_stat = MusaMemcpyAsyncD2H(dst, src, bytes, stream_handle_);
    if (m_stat != mStatus::SUCCESS) {
      done(errors::Internal("MUSA D2H async copy failed."));
      return;
    }
  }
  done(Status::OK());
}

MusaDevice::MusaDevice(Env* env, const DeviceAttributes& attributes,
                       int device_id,
                       ::stream_executor::StreamExecutor* executor)
    : Device(env, attributes), device_id_(device_id) {
  musaSetDevice(device_id_);

  musaError_t stream_err = musaStreamCreate(&stream_);
  if (stream_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create stream: "
               << musaGetErrorString(stream_err);
    stream_ = nullptr;
    device_context_ = nullptr;
    musa_allocator_ = nullptr;
    return;
  }

  mudnn_handle_.reset(new ::musa::dnn::Handle());
  ::musa::dnn::Status s = mudnn_handle_->SetStream(stream_);
  if (s != ::musa::dnn::Status::SUCCESS) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to bind muDNN handle!";
    mudnn_handle_.reset();
    musaStreamDestroy(stream_);
    stream_ = nullptr;
    device_context_ = nullptr;
    musa_allocator_ = nullptr;
    return;
  }

  mublasStatus_t blas_err = mublasCreate(&mublas_handle_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create muBLAS handle!";
    mublas_handle_ = nullptr;
    if (stream_) {
      musaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    mudnn_handle_.reset();
    device_context_ = nullptr;
    musa_allocator_ = nullptr;
    return;
  }

  blas_err = mublasSetStream(mublas_handle_, stream_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to set muBLAS stream!";
    mublasDestroy(mublas_handle_);
    mublas_handle_ = nullptr;
    if (stream_) {
      musaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    mudnn_handle_.reset();
    device_context_ = nullptr;
    musa_allocator_ = nullptr;
    return;
  }

  device_context_ = new MusaDeviceContext(stream_, executor);
  musa_allocator_ = new MusaBFCAllocator(device_id_);

  gpu_device_info_.stream = device_context_->stream();
  gpu_device_info_.default_context = device_context_;
  gpu_device_info_.gpu_id = device_id_;

  set_tensorflow_gpu_device_info(&gpu_device_info_);

  VLOG(1) << ">>> [MUSA] Device " << device_id_ << " initialized successfully";
}

MusaDevice::~MusaDevice() {
  musaSetDevice(device_id_);
  if (device_context_) {
    device_context_->Unref();
  }
  if (mublas_handle_) {
    mublasDestroy(mublas_handle_);
  }
  if (musa_allocator_) {
    delete musa_allocator_;
  }
  if (stream_) {
    musaStreamDestroy(stream_);
  }
}

Allocator* MusaDevice::GetAllocator(AllocatorAttributes attr) {
  return attr.on_host() ? cpu_allocator() : musa_allocator_;
}

Status MusaDevice::Sync() {
  musaSetDevice(device_id_);
  musaError_t err = musaStreamSynchronize(stream_);
  return (err == musaSuccess) ? Status::OK()
                              : errors::Internal("MUSA Device Sync Failed");
}

Status MusaDevice::TryGetDeviceContext(DeviceContext** out_context) {
  if (device_context_) {
    *out_context = device_context_;
    device_context_->Ref();
    return Status::OK();
  }
  return errors::Internal("MusaDeviceContext is null");
}

}  // namespace musa
}  // namespace tensorflow
