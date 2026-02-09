#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_DEVICE_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_DEVICE_H_

#pragma once
#include <mudnn.h>
#include <mublas.h>
#include <musa_runtime.h>
#include <memory>
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/stream_executor/stream.h"
#include "musa_stream.h"
#include "mudnn_base.h" 

namespace tensorflow {
namespace musa {

class MusaDeviceContext : public DeviceContext {
 public:
  explicit MusaDeviceContext(musaStream_t stream, ::stream_executor::StreamExecutor* executor);
  ~MusaDeviceContext() override;

  ::stream_executor::Stream* stream() const override { return official_stream_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;

 private:
  musaStream_t stream_handle_;
  ::stream_executor::internal::StreamInterface* implementation_;
  ::stream_executor::Stream* official_stream_;
};

class MusaDevice : public Device {
 public:
  MusaDevice(Env* env, const DeviceAttributes& attributes, int device_id,
             ::stream_executor::StreamExecutor* executor); 
  ~MusaDevice() override;

  const GpuDeviceInfo* tensorflow_gpu_device_info() const override { return &gpu_device_info_; }
  Status TryGetDeviceContext(DeviceContext** out_context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status Sync() override;

  musaStream_t GetStream() const { return stream_; }
  int get_device_id() const { return device_id_; }

  // 解引用返回句柄
  ::musa::dnn::Handle& mudnn_handle() { return *mudnn_handle_; }
  mublasHandle_t mublas_handle() { return mublas_handle_; }

  ::musa::dnn::MemoryMaintainer GetMemMaintainer(
      std::function<::musa::dnn::MemoryHandler(size_t)> func) {
    return func; 
  }

 private:
  int device_id_;
  musaStream_t stream_;
  MusaDeviceContext* device_context_;
  Allocator* musa_allocator_;
  GpuDeviceInfo gpu_device_info_;

  // 使用智能指针以延迟初始化
  std::unique_ptr<::musa::dnn::Handle> mudnn_handle_;
  mublasHandle_t mublas_handle_;
};

} // namespace musa
} // namespace tensorflow

#endif
