#include "musa_device.h"

#include <cstring>
#include <iostream>

#include "mu/device/musa_event.h"
#include "musa_allocator.h"
#include "musa_event_mgr.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace musa {

MusaDeviceContext::MusaDeviceContext(
    musaStream_t stream, musaStream_t h2d_stream, musaStream_t d2h_stream,
    ::stream_executor::StreamExecutor* executor, MusaEventMgr* event_mgr)
    : stream_handle_(stream),
      h2d_stream_(h2d_stream),
      d2h_stream_(d2h_stream),
      event_mgr_(event_mgr) {
  implementation_ = new ::stream_executor::musa::MusaStream(stream);
  official_stream_ = new ::stream_executor::Stream(executor, implementation_);
  official_stream_->Init();
}

void MusaDeviceContext::ThenExecute(musaStream_t stream,
                                    std::function<void()> func) {
  if (event_mgr_) {
    event_mgr_->ThenExecute(stream, std::move(func));
  } else {
    func();
  }
}

MusaDeviceContext::~MusaDeviceContext() {
  if (official_stream_) {
    official_stream_->BlockHostUntilDone().IgnoreError();
    delete official_stream_;
  }
}

void MusaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                              Device* device,
                                              Tensor* device_tensor,
                                              StatusCallback done,
                                              bool sync_dst_compute) const {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  int device_id = musa_dev->get_device_id();
  musaSetDevice(device_id);

  const void* src = cpu_tensor->tensor_data().data();
  void* dst = const_cast<char*>(device_tensor->tensor_data().data());
  size_t bytes = cpu_tensor->TotalBytes();

  if (bytes == 0) {
    done(Status::OK());
    return;
  }

  // Check if source memory is pinned (musaMemoryTypeHost)
  musaPointerAttributes attributes;
  musaError_t attr_err = musaPointerGetAttributes(&attributes, src);
  bool is_pinned =
      (attr_err == musaSuccess && attributes.type == musaMemoryTypeHost);

  // Clear error state if pointer query failed (expected for pageable memory)
  if (attr_err != musaSuccess) {
    musaGetLastError();
  }

  if (is_pinned) {
    // Fast path: pinned memory can use direct async copy
    musaError_t err =
        musaMemcpyAsync(dst, src, bytes, musaMemcpyHostToDevice, h2d_stream_);
    if (err != musaSuccess) {
      done(errors::Internal("MUSA H2D async copy failed"));
      return;
    }

    if (sync_dst_compute) {
      musaEvent_t copy_done_event;
      musaEventCreateWithFlags(&copy_done_event, musaEventDisableTiming);
      musaEventRecord(copy_done_event, h2d_stream_);
      musaStreamWaitEvent(stream_handle_, copy_done_event, 0);
      musaEventDestroy(copy_done_event);
    }

    if (event_mgr_) {
      event_mgr_->ThenExecute(h2d_stream_, [device_id, done]() {
        musaSetDevice(device_id);
        done(Status::OK());
      });
    } else {
      done(Status::OK());
    }
  } else {
    // Use bounce buffer for pageable memory
    // Small copies (<1KB) use sync path to avoid async overhead and potential
    // driver instability with small async transfers
    const size_t kSmallCopyThreshold = 1024;
    if (bytes <= kSmallCopyThreshold) {
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA H2D small sync copy failed"));
        return;
      }
      done(Status::OK());
      return;
    }

    // Use GPUPinnedMemoryPool for bounce buffer allocation
    // This ensures memory addresses are not reused until GPU async copies
    // complete
    void* bounce_buffer = musa_dev->pinned_memory_pool()->Allocate(bytes);
    if (bounce_buffer == nullptr) {
      LOG(WARNING)
          << "PinnedMemoryPool allocation failed, falling back to sync copy";
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyHostToDevice);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA H2D sync copy failed"));
        return;
      }
      done(Status::OK());
      return;
    }

    // Stage 1: Copy from pageable to pinned (CPU-side, synchronous)
    std::memcpy(bounce_buffer, src, bytes);

    // Stage 2: Async copy from pinned to GPU
    musaError_t err = musaMemcpyAsync(dst, bounce_buffer, bytes,
                                      musaMemcpyHostToDevice, h2d_stream_);
    if (err != musaSuccess) {
      LOG(ERROR) << "MUSA H2D async copy failed: " << musaGetErrorString(err)
                 << " dst=" << dst << " bounce_buffer=" << bounce_buffer
                 << " bytes=" << bytes << " stream=" << h2d_stream_;
      musaFreeHost(bounce_buffer);
      done(errors::Internal("MUSA H2D async copy via bounce buffer failed"));
      return;
    }

    // Setup stream dependency if needed
    if (sync_dst_compute) {
      musaEvent_t copy_done_event;
      musaEventCreateWithFlags(&copy_done_event, musaEventDisableTiming);
      musaEventRecord(copy_done_event, h2d_stream_);
      musaStreamWaitEvent(stream_handle_, copy_done_event, 0);
      musaEventDestroy(copy_done_event);
    }

    // Free bounce buffer - will be returned to pool after GPU copy completes
    musa_dev->pinned_memory_pool()->FreeAsync(bounce_buffer, bytes,
                                              h2d_stream_);

    if (event_mgr_) {
      event_mgr_->ThenExecute(h2d_stream_, [device_id, done]() {
        musaSetDevice(device_id);
        done(Status::OK());
      });
    } else {
      musaStreamSynchronize(h2d_stream_);
      done(Status::OK());
    }
  }
}

void MusaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              StringPiece tensor_name,
                                              Device* device,
                                              Tensor* cpu_tensor,
                                              StatusCallback done) {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  int device_id = musa_dev->get_device_id();
  musaSetDevice(device_id);

  const void* src = device_tensor->tensor_data().data();
  void* dst = const_cast<char*>(cpu_tensor->tensor_data().data());
  size_t bytes = device_tensor->TotalBytes();

  if (bytes > cpu_tensor->TotalBytes()) {
    bytes = cpu_tensor->TotalBytes();
  }
  if (bytes == 0) {
    done(Status::OK());
    return;
  }

  // Check if destination memory is pinned
  musaPointerAttributes attributes;
  musaError_t attr_err = musaPointerGetAttributes(&attributes, dst);
  bool is_pinned =
      (attr_err == musaSuccess && attributes.type == musaMemoryTypeHost);

  // Clear error state if pointer query failed
  if (attr_err != musaSuccess) {
    musaGetLastError();
  }

  if (is_pinned) {
    // Fast path: async copy to pinned memory
    musaEvent_t compute_done_event;
    musaEventCreateWithFlags(&compute_done_event, musaEventDisableTiming);
    musaEventRecord(compute_done_event, stream_handle_);
    musaStreamWaitEvent(d2h_stream_, compute_done_event, 0);
    musaEventDestroy(compute_done_event);

    musaError_t err =
        musaMemcpyAsync(dst, src, bytes, musaMemcpyDeviceToHost, d2h_stream_);
    if (err != musaSuccess) {
      done(errors::Internal("MUSA D2H async copy failed"));
      return;
    }

    if (event_mgr_) {
      event_mgr_->ThenExecute(d2h_stream_, [device_id, done]() {
        musaSetDevice(device_id);
        done(Status::OK());
      });
    } else {
      done(Status::OK());
    }
  } else {
    // Use bounce buffer for pageable memory
    const size_t kSmallCopyThreshold = 1024;
    if (bytes <= kSmallCopyThreshold) {
      musaStreamSynchronize(stream_handle_);
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyDeviceToHost);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA D2H small sync copy failed"));
        return;
      }
      done(Status::OK());
      return;
    }

    void* bounce_buffer = musa_dev->pinned_memory_pool()->Allocate(bytes);
    if (bounce_buffer == nullptr) {
      LOG(WARNING)
          << "PinnedMemoryPool allocation failed, falling back to sync copy";
      musaStreamSynchronize(stream_handle_);
      musaError_t err = musaMemcpy(dst, src, bytes, musaMemcpyDeviceToHost);
      if (err != musaSuccess) {
        done(errors::Internal("MUSA D2H sync copy failed"));
        return;
      }
      done(Status::OK());
      return;
    }

    // Wait for compute stream before starting D2H copy
    musaEvent_t compute_done_event;
    musaEventCreateWithFlags(&compute_done_event, musaEventDisableTiming);
    musaEventRecord(compute_done_event, stream_handle_);
    musaStreamWaitEvent(d2h_stream_, compute_done_event, 0);
    musaEventDestroy(compute_done_event);

    // Stage 1: Async copy from GPU to pinned
    musaError_t err = musaMemcpyAsync(bounce_buffer, src, bytes,
                                      musaMemcpyDeviceToHost, d2h_stream_);
    if (err != musaSuccess) {
      LOG(ERROR) << "MUSA D2H async copy failed: " << musaGetErrorString(err)
                 << " bounce_buffer=" << bounce_buffer << " src=" << src
                 << " bytes=" << bytes << " stream=" << d2h_stream_;
      musaFreeHost(bounce_buffer);
      done(errors::Internal("MUSA D2H async copy to bounce buffer failed"));
      return;
    }

    // Stage 2: Copy from pinned to pageable (CPU-side)
    // FreeAsync ensures bounce_buffer is returned to pool after GPU copy
    // completes
    if (event_mgr_) {
      event_mgr_->ThenExecute(d2h_stream_, [musa_dev, device_id, dst,
                                            bounce_buffer, bytes, done]() {
        musaSetDevice(device_id);
        std::memcpy(dst, bounce_buffer, bytes);
        musa_dev->pinned_memory_pool()->FreeAsync(bounce_buffer, bytes,
                                                  nullptr);
        done(Status::OK());
      });
    } else {
      musaStreamSynchronize(d2h_stream_);
      std::memcpy(dst, bounce_buffer, bytes);
      musa_dev->pinned_memory_pool()->FreeAsync(bounce_buffer, bytes, nullptr);
      done(Status::OK());
    }
  }
}

MusaDevice::MusaDevice(Env* env, const DeviceAttributes& attributes,
                       int device_id,
                       ::stream_executor::StreamExecutor* executor)
    : Device(env, attributes), device_id_(device_id) {
  musaSetDevice(device_id_);

  // Create main compute stream
  musaError_t stream_err = musaStreamCreate(&stream_);
  if (stream_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create stream";
    return;
  }

  // Create H2D stream for host to device transfers
  musaError_t h2d_err = musaStreamCreate(&h2d_stream_);
  if (h2d_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create h2d_stream";
    musaStreamDestroy(stream_);
    return;
  }

  // Create D2H stream for device to host transfers
  musaError_t d2h_err = musaStreamCreate(&d2h_stream_);
  if (d2h_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create d2h_stream";
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    return;
  }

  mudnn_handle_.reset(new ::musa::dnn::Handle());
  ::musa::dnn::Status s = mudnn_handle_->SetStream(stream_);
  if (s != ::musa::dnn::Status::SUCCESS) {
    mudnn_handle_.reset();
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    musaStreamDestroy(d2h_stream_);
    return;
  }

  mublasStatus_t blas_err = mublasCreate(&mublas_handle_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    mublas_handle_ = nullptr;
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    musaStreamDestroy(d2h_stream_);
    mudnn_handle_.reset();
    return;
  }

  blas_err = mublasSetStream(mublas_handle_, stream_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    mublasDestroy(mublas_handle_);
    mublas_handle_ = nullptr;
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    musaStreamDestroy(d2h_stream_);
    mudnn_handle_.reset();
    return;
  }

  // Create Event Manager for async event polling
  event_mgr_ = new MusaEventMgr(device_id_);

  // Pass streams to Context (compute, h2d, d2h)
  device_context_ = new MusaDeviceContext(stream_, h2d_stream_, d2h_stream_,
                                          executor, event_mgr_);

  // Get total device memory
  size_t total_memory = 0, free_memory = 0;
  musaMemGetInfo(&free_memory, &total_memory);

  // Use TensorFlow's official BFCAllocator with MusaSubAllocator
  // Note: allow_growth=false to pre-allocate a large chunk upfront
  // garbage_collection=true to reclaim unused memory
  // Use 90% of free memory to leave headroom for driver
  size_t bfc_memory_limit = static_cast<size_t>(free_memory * 0.9);

  musa_allocator_ = new BFCAllocator(new MusaSubAllocator(device_id_, {}, {}),
                                     bfc_memory_limit,
                                     false,  // allow_growth
                                     "Musa_BFC_Allocator",
                                     true  // garbage_collection
  );

  VLOG(1) << ">>> [MUSA] Device " << device_id_
          << " using official TF BFCAllocator with bfc_memory_limit="
          << bfc_memory_limit << " bytes (free_memory=" << free_memory << ")";

  // Initialize Host Pinned Memory Allocator (BFCAllocator - kept for
  // compatibility)
  musa_host_allocator_ =
      new BFCAllocator(new MusaHostSubAllocator({}, {}),
                       256ULL * 1024 * 1024,  // 256 MB
                       true, "Musa_Host_BFC_Allocator", true);

  // Initialize GPUPinnedMemoryPool for bounce buffer management
  // This pool ensures memory addresses are not reused until GPU async copies
  // complete
  pinned_memory_pool_ = new GPUPinnedMemoryPool(device_id_);

  VLOG(1) << ">>> [MUSA] Initialized GPUPinnedMemoryPool for Bounce Buffers";

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
  if (pinned_memory_pool_) {
    delete pinned_memory_pool_;
  }
  if (musa_host_allocator_) {
    delete musa_host_allocator_;
  }
  if (musa_allocator_) {
    delete musa_allocator_;
  }
  if (event_mgr_) {
    delete event_mgr_;
  }
  if (d2h_stream_) {
    musaStreamDestroy(d2h_stream_);
  }
  if (h2d_stream_) {
    musaStreamDestroy(h2d_stream_);
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
  musaError_t err = musaDeviceSynchronize();
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
