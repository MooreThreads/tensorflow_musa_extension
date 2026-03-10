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

#include "mu/device/musa_executor.h"

#include <musa_runtime.h>

#include "mu/device/musa_event.h"
#include "mu/device/musa_stream.h"
#include "mu/device/musa_timer.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/timer.h"

namespace stream_executor {
namespace musa {

// Helper function to convert MUSA error to port::Status
port::Status FromMusaStatus(musaError_t err) {
  if (err == musaSuccess) {
    return port::Status::OK();
  }
  return port::Status(port::error::INTERNAL,
                      absl::StrCat("MUSA operation failed: ",
                                   musaGetErrorString(err)));
}

MusaExecutor::MusaExecutor(const PluginConfig& plugin_config)
    : plugin_config_(plugin_config), device_ordinal_(0), device_count_(0) {}

MusaExecutor::~MusaExecutor() {}

port::Status MusaExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
  device_ordinal_ = device_ordinal;
  musaError_t err = musaGetDeviceCount(&device_count_);
  if (err != musaSuccess) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("musaGetDeviceCount failed: ",
                                     musaGetErrorString(err)));
  }
  if (device_ordinal >= device_count_) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        absl::StrCat("Invalid device ordinal ", device_ordinal,
                                     " (only ", device_count_, " devices available)"));
  }
  return port::Status::OK();
}

DeviceMemoryBase MusaExecutor::Allocate(uint64 size, int64 memory_space) {
  musaSetDevice(device_ordinal_);
  void* ptr = nullptr;
  musaError_t err = musaMalloc(&ptr, size);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMalloc failed: " << musaGetErrorString(err)
               << " (size=" << size << ")";
    return DeviceMemoryBase(nullptr, 0);
  }
  return DeviceMemoryBase(ptr, size);
}

void* MusaExecutor::GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                                 uint64 size) {
  if (parent == nullptr || parent->opaque() == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<char*>(parent->opaque()) + offset;
}

void MusaExecutor::Deallocate(DeviceMemoryBase* mem) {
  if (mem && mem->opaque()) {
    musaSetDevice(device_ordinal_);
    musaError_t err = musaFree(mem->opaque());
    if (err != musaSuccess) {
      LOG(ERROR) << "musaFree failed: " << musaGetErrorString(err);
    }
  }
}

void* MusaExecutor::HostMemoryAllocate(uint64 size) {
  void* ptr = nullptr;
  musaError_t err = musaMallocHost(&ptr, size);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMallocHost failed: " << musaGetErrorString(err);
    return nullptr;
  }
  return ptr;
}

void MusaExecutor::HostMemoryDeallocate(void* mem) {
  if (mem) {
    musaError_t err = musaFreeHost(mem);
    if (err != musaSuccess) {
      LOG(ERROR) << "musaFreeHost failed: " << musaGetErrorString(err);
    }
  }
}

bool MusaExecutor::HostMemoryRegister(void* mem, uint64 size) {
  // MUSA doesn't require explicit host memory registration like CUDA
  return true;
}

bool MusaExecutor::HostMemoryUnregister(void* mem) {
  // MUSA doesn't require explicit host memory unregistration like CUDA
  return true;
}

bool MusaExecutor::SynchronizeAllActivity() {
  musaSetDevice(device_ordinal_);
  musaError_t err = musaDeviceSynchronize();
  if (err != musaSuccess) {
    LOG(ERROR) << "musaDeviceSynchronize failed: " << musaGetErrorString(err);
    return false;
  }
  return true;
}

port::Status MusaExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64 size) {
  if (location == nullptr || location->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid memory location");
  }
  musaSetDevice(device_ordinal_);
  return FromMusaStatus(musaMemset(location->opaque(), 0, size));
}

port::Status MusaExecutor::SynchronousMemSet(DeviceMemoryBase* location, int value,
                                             uint64 size) {
  if (location == nullptr || location->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid memory location");
  }
  musaSetDevice(device_ordinal_);
  return FromMusaStatus(musaMemset(location->opaque(), value, size));
}

port::Status MusaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src, uint64 size) {
  if (gpu_dst == nullptr || gpu_dst->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid device memory");
  }
  if (host_src == nullptr && size > 0) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid host memory");
  }
  musaSetDevice(device_ordinal_);
  return FromMusaStatus(musaMemcpy(gpu_dst->opaque(), host_src, size,
                                   musaMemcpyHostToDevice));
}

port::Status MusaExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64 size) {
  if (host_dst == nullptr && size > 0) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid host memory");
  }
  if (gpu_src.opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid device memory");
  }
  musaSetDevice(device_ordinal_);
  return FromMusaStatus(musaMemcpy(host_dst, gpu_src.opaque(), size,
                                   musaMemcpyDeviceToHost));
}

port::Status MusaExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) {
  if (gpu_dst == nullptr || gpu_dst->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid destination memory");
  }
  if (gpu_src.opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid source memory");
  }
  musaSetDevice(device_ordinal_);
  return FromMusaStatus(musaMemcpy(gpu_dst->opaque(), gpu_src.opaque(), size,
                                   musaMemcpyDeviceToDevice));
}

port::Status MusaExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                   uint64 size) {
  if (location == nullptr || location->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid memory location");
  }
  musaStream_t musa_stream = GetMusaStream(stream);
  return FromMusaStatus(musaMemsetAsync(location->opaque(), 0, size, musa_stream));
}

port::Status MusaExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                  uint8 pattern, uint64 size) {
  if (location == nullptr || location->opaque() == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid memory location");
  }
  musaStream_t musa_stream = GetMusaStream(stream);
  return FromMusaStatus(
      musaMemsetAsync(location->opaque(), pattern, size, musa_stream));
}

port::Status MusaExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                    uint32 pattern, uint64 size) {
  // MUSA doesn't have a native 32-bit memset, so we use 8-bit with lower byte
  return Memset(stream, location, static_cast<uint8>(pattern & 0xFF), size);
}

bool MusaExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64 size) {
  musaStream_t musa_stream = GetMusaStream(stream);
  musaError_t err =
      musaMemcpyAsync(host_dst, gpu_src.opaque(), size,
                      musaMemcpyDeviceToHost, musa_stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMemcpyAsync D2H failed: " << musaGetErrorString(err);
    return false;
  }
  return true;
}

bool MusaExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64 size) {
  musaStream_t musa_stream = GetMusaStream(stream);
  musaError_t err =
      musaMemcpyAsync(gpu_dst->opaque(), host_src, size,
                      musaMemcpyHostToDevice, musa_stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMemcpyAsync H2D failed: " << musaGetErrorString(err);
    return false;
  }
  return true;
}

bool MusaExecutor::MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64 size) {
  musaStream_t musa_stream = GetMusaStream(stream);
  musaError_t err =
      musaMemcpyAsync(gpu_dst->opaque(), gpu_src.opaque(), size,
                      musaMemcpyDeviceToDevice, musa_stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMemcpyAsync D2D failed: " << musaGetErrorString(err);
    return false;
  }
  return true;
}

bool MusaExecutor::HostCallback(Stream* stream, std::function<void()> callback) {
  // MUSA doesn't support host callbacks directly via the runtime API
  // For now, we execute the callback immediately and return true
  // A full implementation would use a separate thread to handle callbacks
  callback();
  return true;
}

bool MusaExecutor::HostCallback(Stream* stream,
                                std::function<port::Status()> callback) {
  // MUSA doesn't support host callbacks directly via the runtime API
  // For now, we execute the callback immediately and return true
  // A full implementation would use a separate thread to handle callbacks
  port::Status status = callback();
  return status.ok();
}

bool MusaExecutor::AllocateStream(Stream* stream) {
  // Stream allocation is handled in GetStreamImplementation
  return true;
}

void MusaExecutor::DeallocateStream(Stream* stream) {
  // Stream deallocation is handled in MusaStream destructor
}

bool MusaExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  musaStream_t dependent_stream = GetMusaStream(dependent);
  musaStream_t other_stream = GetMusaStream(other);
  
  // Create an event on the other stream
  musaEvent_t event;
  musaError_t err = musaEventCreate(&event);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaEventCreate failed: " << musaGetErrorString(err);
    return false;
  }
  
  err = musaEventRecord(event, other_stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaEventRecord failed: " << musaGetErrorString(err);
    musaEventDestroy(event);
    return false;
  }
  
  err = musaStreamWaitEvent(dependent_stream, event, 0);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaStreamWaitEvent failed: " << musaGetErrorString(err);
    musaEventDestroy(event);
    return false;
  }
  
  musaEventDestroy(event);
  return true;
}

bool MusaExecutor::AllocateTimer(Timer* timer) {
  // Timer allocation is handled in GetTimerImplementation
  return true;
}

void MusaExecutor::DeallocateTimer(Timer* timer) {
  // Timer deallocation is handled in MusaTimer destructor
}

bool MusaExecutor::StartTimer(Stream* stream, Timer* timer) {
  MusaTimer* musa_timer = static_cast<MusaTimer*>(timer->implementation());
  if (!musa_timer) {
    return false;
  }
  return musa_timer->Start(GetMusaStream(stream));
}

bool MusaExecutor::StopTimer(Stream* stream, Timer* timer) {
  MusaTimer* musa_timer = static_cast<MusaTimer*>(timer->implementation());
  if (!musa_timer) {
    return false;
  }
  return musa_timer->Stop(GetMusaStream(stream));
}

port::Status MusaExecutor::BlockHostUntilDone(Stream* stream) {
  musaStream_t musa_stream = GetMusaStream(stream);
  return FromMusaStatus(musaStreamSynchronize(musa_stream));
}

int MusaExecutor::PlatformDeviceCount() { return device_count_; }

port::Status MusaExecutor::EnablePeerAccessTo(StreamExecutorInterface* other) {
  // MUSA may not support peer access like CUDA, return OK for compatibility
  return port::Status::OK();
}

bool MusaExecutor::CanEnablePeerAccessTo(StreamExecutorInterface* other) {
  // MUSA may not support peer access like CUDA
  return false;
}

bool MusaExecutor::DeviceMemoryUsage(int64* free, int64* total) const {
  musaSetDevice(device_ordinal_);
  size_t free_val, total_val;
  musaError_t err = musaMemGetInfo(&free_val, &total_val);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaMemGetInfo failed: " << musaGetErrorString(err);
    return false;
  }
  if (free) *free = static_cast<int64>(free_val);
  if (total) *total = static_cast<int64>(total_val);
  return true;
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription() const {
  return CreateDeviceDescription(device_ordinal_);
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription(int device_ordinal) {
  musaSetDevice(device_ordinal);
  
  internal::DeviceDescriptionBuilder builder;
  
  // Get device properties
  musaDeviceProp props;
  musaError_t err = musaGetDeviceProperties(&props, device_ordinal);
  if (err != musaSuccess) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("musaGetDeviceProperties failed: ",
                                     musaGetErrorString(err)));
  }
  
  builder.set_name(props.name);
  builder.set_platform_version(
      absl::StrCat("MUSA Compute Capability ", props.major, ".", props.minor));
  builder.set_driver_version("");
  builder.set_runtime_version("");
  builder.set_pci_bus_id("");
  
  // Memory information
  size_t free_memory, total_memory;
  err = musaMemGetInfo(&free_memory, &total_memory);
  if (err == musaSuccess) {
    builder.set_device_memory_size(static_cast<int64>(total_memory));
    builder.set_memory_bandwidth(0);  // Not available from MUSA runtime
  }
  
  // Clock rates
  builder.set_clock_rate_ghz(static_cast<float>(props.clockRate) / 1000000.0f);
  
  // Compute capability
  builder.set_cuda_compute_capability(props.major, props.minor);
  
  // Core counts
  builder.set_core_count(props.multiProcessorCount);
  builder.set_ecc_enabled(false);
  
  // NUMA node (default to 0 if not available)
  builder.set_numa_node(0);
  
  return builder.Build();
}

port::Status MusaExecutor::AllocateEvent(Event* event) {
  auto* musa_event = static_cast<MusaEvent*>(event->implementation());
  if (!musa_event) {
    return port::Status(port::error::INTERNAL, "Invalid event implementation");
  }
  return musa_event->Init() ? port::Status::OK()
                            : port::Status(port::error::INTERNAL,
                                           "Failed to initialize MUSA event");
}

port::Status MusaExecutor::DeallocateEvent(Event* event) {
  auto* musa_event = static_cast<MusaEvent*>(event->implementation());
  if (musa_event) {
    musa_event->Destroy();
  }
  return port::Status::OK();
}

port::Status MusaExecutor::RecordEvent(Stream* stream, Event* event) {
  auto* musa_event = static_cast<MusaEvent*>(event->implementation());
  if (!musa_event || !musa_event->IsInitialized()) {
    return port::Status(port::error::INTERNAL, "Invalid event");
  }
  musaStream_t musa_stream = GetMusaStream(stream);
  return FromMusaStatus(musaEventRecord(musa_event->handle(), musa_stream));
}

port::Status MusaExecutor::WaitForEvent(Stream* stream, Event* event) {
  auto* musa_event = static_cast<MusaEvent*>(event->implementation());
  if (!musa_event || !musa_event->IsInitialized()) {
    return port::Status(port::error::INTERNAL, "Invalid event");
  }
  musaStream_t musa_stream = GetMusaStream(stream);
  return FromMusaStatus(musaStreamWaitEvent(musa_stream, musa_event->handle(), 0));
}

Event::Status MusaExecutor::PollForEventStatus(Event* event) {
  auto* musa_event = static_cast<MusaEvent*>(event->implementation());
  if (!musa_event || !musa_event->IsInitialized()) {
    return Event::Status::kError;
  }
  musaError_t err = musaEventQuery(musa_event->handle());
  if (err == musaSuccess) return Event::Status::kComplete;
  if (err == musaErrorNotReady) return Event::Status::kPending;
  return Event::Status::kError;
}

std::unique_ptr<internal::EventInterface> MusaExecutor::CreateEventImplementation() {
  return std::make_unique<MusaEvent>();
}

std::unique_ptr<internal::KernelInterface> MusaExecutor::CreateKernelImplementation() {
  return nullptr;  // Not implemented
}

std::unique_ptr<internal::StreamInterface> MusaExecutor::GetStreamImplementation() {
  musaStream_t stream;
  musaError_t err = musaStreamCreate(&stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "musaStreamCreate failed: " << musaGetErrorString(err);
    return nullptr;
  }
  return std::make_unique<MusaStream>(stream);
}

std::unique_ptr<internal::TimerInterface> MusaExecutor::GetTimerImplementation() {
  return std::make_unique<MusaTimer>();
}

musaStream_t MusaExecutor::GetMusaStream(Stream* stream) {
  if (stream == nullptr) {
    return 0;  // Default stream
  }
  auto* musa_stream = static_cast<MusaStream*>(stream->implementation());
  return musa_stream ? musa_stream->handle() : 0;
}

}  // namespace musa
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(musa_executor, {});
