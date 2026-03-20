#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_DEVICE_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_DEVICE_H_

#pragma once
#include <mublas.h>
#include <mudnn.h>
#include <musa_runtime.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "mudnn_base.h"
#include "musa_allocator.h"
#include "musa_event_mgr.h"
#include "musa_host_allocator.h"
#include "musa_stream.h"
#include "pinned_memory_pool.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace musa {

struct AsyncCopyPayload {
  StatusCallback done;
  musaEvent_t sync_event;
};

class MusaDeviceContext : public DeviceContext {
 public:
  explicit MusaDeviceContext(musaStream_t stream, musaStream_t h2d_stream,
                             musaStream_t d2h_stream,
                             ::stream_executor::StreamExecutor* executor,
                             MusaEventMgr* event_mgr);
  ~MusaDeviceContext() override;

  ::stream_executor::Stream* stream() const override {
    return official_stream_;
  }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;

  void ThenExecute(musaStream_t stream, std::function<void()> func);

  MusaEventMgr* event_mgr() const { return event_mgr_; }

 private:
  musaStream_t stream_handle_;
  musaStream_t h2d_stream_;
  musaStream_t d2h_stream_;
  ::stream_executor::internal::StreamInterface* implementation_;
  ::stream_executor::Stream* official_stream_;
  MusaEventMgr* event_mgr_;
};

class MusaDevice : public Device {
 public:
  MusaDevice(Env* env, const DeviceAttributes& attributes, int device_id,
             ::stream_executor::StreamExecutor* executor);
  ~MusaDevice() override;

  const GpuDeviceInfo* tensorflow_gpu_device_info() const override {
    return &gpu_device_info_;
  }
  Status TryGetDeviceContext(DeviceContext** out_context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status Sync() override;

  musaStream_t GetStream() const { return stream_; }
  int get_device_id() const { return device_id_; }
  Allocator* musa_host_allocator() const { return musa_host_allocator_; }

  ::musa::dnn::Handle& mudnn_handle() { return *mudnn_handle_; }
  mublasHandle_t mublas_handle() { return mublas_handle_; }
  MusaEventMgr* event_mgr() const { return event_mgr_; }

  ::musa::dnn::MemoryMaintainer GetMemMaintainer(
      std::function<::musa::dnn::MemoryHandler(size_t)> func) {
    return func;
  }

  GPUPinnedMemoryPool* pinned_memory_pool() const {
    return pinned_memory_pool_;
  }

 private:
  int device_id_;
  musaStream_t stream_;
  musaStream_t h2d_stream_;
  musaStream_t d2h_stream_;
  MusaDeviceContext* device_context_;
  Allocator* musa_allocator_;
  Allocator* musa_host_allocator_;
  GPUPinnedMemoryPool* pinned_memory_pool_;
  GpuDeviceInfo gpu_device_info_;
  MusaEventMgr* event_mgr_;

  std::unique_ptr<::musa::dnn::Handle> mudnn_handle_;
  mublasHandle_t mublas_handle_;
};

}  // namespace musa
}  // namespace tensorflow

#endif
