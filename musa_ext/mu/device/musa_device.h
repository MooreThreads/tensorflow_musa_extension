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
#include "musa_stream.h"
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
                             ::stream_executor::StreamExecutor* executor);
  ~MusaDeviceContext() override;

  ::stream_executor::Stream* stream() const override {
    return official_stream_;
  }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;

 private:
  musaStream_t stream_handle_;
  musaStream_t h2d_stream_;
  ::stream_executor::internal::StreamInterface* implementation_;
  ::stream_executor::Stream* official_stream_;

  mutable std::mutex cleanup_mu_;
  mutable std::queue<AsyncCopyPayload*> cleanup_queue_;
  mutable std::atomic<bool> stop_polling_{false};
  mutable std::thread polling_thread_;
  void PollingLoop() const;
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

  ::musa::dnn::Handle& mudnn_handle() { return *mudnn_handle_; }
  mublasHandle_t mublas_handle() { return mublas_handle_; }

  ::musa::dnn::MemoryMaintainer GetMemMaintainer(
      std::function<::musa::dnn::MemoryHandler(size_t)> func) {
    return func;
  }

 private:
  int device_id_;
  musaStream_t stream_;
  musaStream_t h2d_stream_;
  MusaDeviceContext* device_context_;
  Allocator* musa_allocator_;
  GpuDeviceInfo gpu_device_info_;

  std::unique_ptr<::musa::dnn::Handle> mudnn_handle_;
  mublasHandle_t mublas_handle_;
};

}  // namespace musa
}  // namespace tensorflow

#endif
