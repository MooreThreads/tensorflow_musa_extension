#include "musa_device.h"

#include <iostream>

#include "mu/device/musa_event.h"
#include "musa_allocator.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace musa {

MusaDeviceContext::MusaDeviceContext(
    musaStream_t stream, ::stream_executor::StreamExecutor* executor)
    : stream_handle_(stream) {
  implementation_ = new ::stream_executor::musa::MusaStream(stream);

  official_stream_ = new ::stream_executor::Stream(executor, implementation_);

  // 初始化 Stream
  official_stream_->Init();
}

MusaDeviceContext::~MusaDeviceContext() {
  if (official_stream_) {
    // Wait for all async operations to complete
    official_stream_->BlockHostUntilDone().IgnoreError();
    delete official_stream_;
    // Note: official_stream_ owns implementation_, so we don't delete it
    // separately to avoid double-free
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
    // Optimization: For small transfers, use synchronous copy which is often
    // faster than async + sync due to lower overhead.
    // For large transfers, use async without immediate sync for better
    // pipelining.
    constexpr size_t kSmallTransferThreshold = 65536;  // 64KB

    if (bytes <= kSmallTransferThreshold) {
      // Synchronous copy for small transfers
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA H2D sync copy failed: ",
                              musaGetErrorString(err)));
        return;
      }
    } else {
      // Async copy for large transfers
      mStatus m_stat = MusaMemcpyAsyncH2D(dst, src, bytes, stream_handle_);
      if (m_stat != mStatus::SUCCESS) {
        done(errors::Internal("MUSA H2D async copy init failed."));
        return;
      }

      // Only synchronize if explicitly requested
      if (sync_dst_compute) {
        musaError_t sync_err = musaStreamSynchronize(stream_handle_);
        if (sync_err != musaSuccess) {
          done(errors::Internal("MUSA H2D stream sync failed: ",
                                musaGetErrorString(sync_err)));
          return;
        }
      }
      // Otherwise, let TensorFlow's stream dependency tracking handle
      // synchronization
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
    // Optimization: For small transfers, use synchronous copy which is often
    // faster than async + sync due to lower overhead.
    // For large transfers, use async without immediate sync for better
    // pipelining.
    constexpr size_t kSmallTransferThreshold = 65536;  // 64KB

    if (bytes <= kSmallTransferThreshold) {
      // Synchronous copy for small transfers
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyDeviceToHost);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA D2H sync copy failed: ",
                              musaGetErrorString(err)));
        return;
      }
    } else {
      // Async copy for large transfers
      mStatus m_stat = MusaMemcpyAsyncD2H(dst, src, bytes, stream_handle_);
      if (m_stat != mStatus::SUCCESS) {
        done(errors::Internal("MUSA D2H async copy init failed."));
        return;
      }
      // For D2H, we typically need to ensure completion for CPU-side
      // consumption. Use lazy sync - TensorFlow's dependency tracking will
      // ensure proper ordering. Only sync if the caller explicitly requests it.
    }
  }
  done(Status::OK());
}

MusaDevice::MusaDevice(Env* env, const DeviceAttributes& attributes,
                       int device_id,
                       ::stream_executor::StreamExecutor* executor)
    : Device(env, attributes), device_id_(device_id) {
  // Set device
  musaSetDevice(device_id_);

  // Create stream
  musaError_t stream_err = musaStreamCreate(&stream_);
  if (stream_err != musaSuccess) {
    std::cerr << ">>> [MUSA] ERROR: Device " << device_id_
              << " failed to create stream: " << musaGetErrorString(stream_err)
              << std::endl;
    stream_ = nullptr;
    device_context_ = nullptr;
    musa_allocator_ = nullptr;
    return;
  }

  // Initialize muDNN handle
  mudnn_handle_.reset(new ::musa::dnn::Handle());
  ::musa::dnn::Status s = mudnn_handle_->SetStream(stream_);
  if (s != ::musa::dnn::Status::SUCCESS) {
    std::cerr << ">>> [MUSA] ERROR: Device " << device_id_
              << " failed to bind muDNN handle! Error code: "
              << static_cast<int>(s) << std::endl;
    mudnn_handle_.reset();
  }

  // Initialize muBLAS handle
  mublasStatus_t blas_err = mublasCreate(&mublas_handle_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    std::cerr << ">>> [MUSA] ERROR: Device " << device_id_
              << " failed to create muBLAS handle! Error code: "
              << static_cast<int>(blas_err) << std::endl;
    mublas_handle_ = nullptr;
  } else {
    mublasSetStream(mublas_handle_, stream_);
  }

  // Initialize Context
  device_context_ = new MusaDeviceContext(stream_, executor);

  // Use BFC allocator for better performance with memory pooling
  musa_allocator_ = new MusaBFCAllocator(device_id_);

  gpu_device_info_.stream = device_context_->stream();
  gpu_device_info_.default_context = device_context_;
  gpu_device_info_.gpu_id = device_id_;

  set_tensorflow_gpu_device_info(&gpu_device_info_);
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
