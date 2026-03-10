/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_

#include <musa_runtime.h>

#include <memory>

#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/timer.h"

namespace stream_executor {
namespace musa {

// Helper function to convert MUSA status to port::Status
port::Status FromMusaStatus(musaError_t err);

// MUSA implementation of the StreamExecutorInterface.
class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  explicit MusaExecutor(const PluginConfig& plugin_config);
  ~MusaExecutor() override;

  port::Status Init(int device_ordinal,
                    DeviceOptions device_options) override;

  port::Status GetKernel(const MultiKernelLoaderSpec& spec,
                         KernelBase* kernel) override {
    return port::UnimplementedError("Not Implemented");
  }

  bool UnloadModule(ModuleHandle module_handle) override { return false; }

  port::Status LoadModule(const MultiModuleLoaderSpec& spec,
                          ModuleHandle* module_handle) override {
    return port::UnimplementedError("Not Implemented");
  }

  port::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const KernelBase& k,
                      const KernelArgsArrayBase& args) override {
    return port::UnimplementedError("Not Implemented");
  }

  void UnloadKernel(const KernelBase* kernel) override {}

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override;

  DeviceMemoryBase Allocate(uint64 size) {
    return Allocate(size, /*memory_space=*/0);
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override;

  void Deallocate(DeviceMemoryBase* mem) override;

  void* UnifiedMemoryAllocate(uint64 size) override { return nullptr; }

  void UnifiedMemoryDeallocate(void* mem) override {}

  void* HostMemoryAllocate(uint64 size) override;

  void HostMemoryDeallocate(void* mem) override;

  bool HostMemoryRegister(void* mem, uint64 size) override;

  bool HostMemoryUnregister(void* mem) override;

  bool SynchronizeAllActivity() override;

  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64 size) override;

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64 size) override;

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64 size) override;

  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64 size) override;

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64 size) override;

  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64 size) override;

  port::Status Memset(Stream* stream, DeviceMemoryBase* location,
                      uint8 pattern, uint64 size) override;

  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64 size) override;

  bool Memcpy(Stream* stream, void* host_dst,
              const DeviceMemoryBase& gpu_src, uint64 size) override;

  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
              const void* host_src, uint64 size) override;

  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64 size) override;

  bool HostCallback(Stream* stream, std::function<void()> callback) override;

  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override;

  port::Status AllocateEvent(Event* event) override;

  port::Status DeallocateEvent(Event* event) override;

  port::Status RecordEvent(Stream* stream, Event* event) override;

  port::Status WaitForEvent(Stream* stream, Event* event) override;

  Event::Status PollForEventStatus(Event* event) override;

  bool AllocateStream(Stream* stream) override;

  void DeallocateStream(Stream* stream) override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  bool AllocateTimer(Timer* timer) override;

  void DeallocateTimer(Timer* timer) override;

  bool StartTimer(Stream* stream, Timer* timer) override;

  bool StopTimer(Stream* stream, Timer* timer) override;

  port::Status BlockHostUntilDone(Stream* stream) override;

  port::Status GetStatus(Stream* stream) override {
    return port::Status(port::error::UNIMPLEMENTED,
                        "GetStatus is not supported on this executor.");
  }

  int PlatformDeviceCount() override;

  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override;

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override;

  int64 GetDeviceLoad() override { return -1; }

  bool DeviceMemoryUsage(int64* free, int64* total) const override;

  bool GetSymbol(const std::string& symbol_name, ModuleHandle module_handle,
                 void** mem, size_t* bytes) override {
    return false;
  }

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  static port::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  bool RegisterTraceListener(TraceListener* listener) override { return false; }

  bool UnregisterTraceListener(TraceListener* listener) override {
    return false;
  }

  bool SupportsBlas() const override { return false; }

  blas::BlasSupport* CreateBlas() override { return nullptr; }

  bool SupportsFft() const override { return false; }

  fft::FftSupport* CreateFft() override { return nullptr; }

  bool SupportsRng() const override { return false; }

  rng::RngSupport* CreateRng() override { return nullptr; }

  bool SupportsDnn() const override { return false; }

  dnn::DnnSupport* CreateDnn() override { return nullptr; }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override;

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override;

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override;

  void* GpuContextHack() override { return nullptr; }

  // Returns the musaStream_t for the given Stream.
  musaStream_t GetMusaStream(Stream* stream);

 private:
  PluginConfig plugin_config_;
  int device_ordinal_;
  int device_count_;

  SE_DISALLOW_COPY_AND_ASSIGN(MusaExecutor);
};

}  // namespace musa
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
