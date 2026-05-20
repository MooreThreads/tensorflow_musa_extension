#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_

#if __has_include("tensorflow/stream_executor/lib/status.h")

#include <memory>

#include "musa_device.h"
#include "musa_event.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "musa_stream.h"
#if __has_include("tensorflow/stream_executor/lib/status.h")
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#else
#include "absl/status/status.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/stream_executor/stream_executor_internal.h"
namespace stream_executor {
namespace port {
using Status = tsl::Status;
template <typename T>
using StatusOr = tsl::StatusOr<T>;
namespace error {
constexpr absl::StatusCode INTERNAL = absl::StatusCode::kInternal;
}  // namespace error
}  // namespace port
}  // namespace stream_executor
#endif
namespace stream_executor {
namespace musa {

inline port::Status FromMusaStatus(mStatus s) {
  if (s == mStatus::SUCCESS) {
    return port::Status::OK();
  }
  return port::Status(port::error::INTERNAL, "MUSA Operation Failed");
}

class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  explicit MusaExecutor(const PluginConfig& plugin_config)
      : plugin_config_(plugin_config) {}
  ~MusaExecutor() override {}

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
    device_ordinal_ = device_ordinal;
    return port::Status::OK();
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    musaStream_t h;
    musaError_t err = musaStreamCreate(&h);
    if (err != musaSuccess) {
      LOG(ERROR) << "musaStreamCreate failed: " << musaGetErrorString(err);
      return nullptr;
    }
    return std::make_unique<MusaStream>(h);
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return std::make_unique<MusaEvent>();
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return nullptr;
  }

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override {
    if (size == 0) {
      return DeviceMemoryBase(nullptr, 0);
    }
    musaSetDevice(device_ordinal_);
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);
    if (err != musaSuccess) {
      LOG(ERROR) << "MusaExecutor::Allocate failed for " << size
                 << " bytes: " << musaGetErrorString(err);
      return DeviceMemoryBase(nullptr, 0);
    }
    return DeviceMemoryBase(ptr, size);
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override {
    return reinterpret_cast<char*>(parent->opaque()) + offset;
  }

  void Deallocate(DeviceMemoryBase* mem) override {
    if (mem && mem->opaque()) {
      musaSetDevice(device_ordinal_);
      musaError_t err = musaFree(mem->opaque());
      if (err != musaSuccess) {
        LOG(ERROR) << "MUSA Deallocate failed: " << musaGetErrorString(err);
      }
    }
  }

  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  void* HostMemoryAllocate(uint64 size) override { return nullptr; }
  void HostMemoryDeallocate(void* mem) override {}

  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64 size) override {
    mHandle h;

    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64 size) override {
    mHandle h;
    return FromMusaStatus(tensorflow::musa::Memset(
        h, location->opaque(), size, static_cast<uint8_t>(value)));
  }

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64 size) override {
    // H2D
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyH2D(gpu_dst->opaque(), host_src, size));
  }

  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64 size) override {
    // D2H
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyD2H(host_dst, gpu_src.opaque(), size));
  }

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64 size) override {
    // D2D
    return FromMusaStatus(tensorflow::musa::MusaMemcpyD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size));
  }

  musaStream_t GetMusaStream(Stream* stream) {
    auto* musa_stream_impl = static_cast<MusaStream*>(stream->implementation());

    return musa_stream_impl->GetStream();
  }

  // D2D Async
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64 size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // H2D Async
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64 size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncH2D(
        gpu_dst->opaque(), host_src, size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // D2H Async
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64 size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2H(
        host_dst, gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // MemZero Async
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64 size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  // Memset32 Async
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64 size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset32(h, location->opaque(), size, pattern));
  }

  port::Status BlockHostUntilDone(Stream* stream) override {
    internal::StreamInterface* implementation = stream->implementation();
    auto* musa_stream = static_cast<MusaStream*>(implementation);
    return musa_stream->BlockHostUntilDone_DEBUG(stream);
  }

  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override {
    // Execute callback asynchronously via a host function
    // This ensures the callback runs after all preceding stream operations
    musaStream_t musa_stream = GetMusaStream(stream);
    musaError_t err = musaLaunchHostFunc(
        musa_stream,
        [](void* user_data) {
          auto* cb = static_cast<std::function<port::Status()>*>(user_data);
          (*cb)();
          delete cb;
        },
        new std::function<port::Status()>(std::move(callback)));
    if (err != musaSuccess) {
      LOG(WARNING) << "MusaExecutor::HostCallback failed: "
                   << musaGetErrorString(err);
      return false;
    }
    return true;
  }

  bool AllocateTimer(Timer* timer) override { return true; }
  void DeallocateTimer(Timer* timer) override {}
  bool StartTimer(Stream* stream, Timer* timer) override { return true; }
  bool StopTimer(Stream* stream, Timer* timer) override { return true; }

  int PlatformDeviceCount() override { return 1; }
  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return port::Status::OK();
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return false;
  }

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    internal::DeviceDescriptionBuilder builder;
    builder.set_name("MUSA Device");
    return builder.Build();
  }

  bool SynchronizeAllActivity() override { return true; }
  bool DeviceMemoryUsage(int64* free, int64* total) const override {
    return false;
  }
  bool AllocateStream(Stream* stream) override { return true; }
  void DeallocateStream(Stream* stream) override {}
  bool CreateStreamDependency(Stream* dependent, Stream* other) override {
    // Create an event on 'other' stream and wait on 'dependent' stream
    musaEvent_t event;
    musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventCreate failed: "
                 << musaGetErrorString(err);
      return false;
    }

    musaStream_t other_stream = GetMusaStream(other);
    err = musaEventRecord(event, other_stream);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventRecord failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    musaStream_t dependent_stream = GetMusaStream(dependent);
    err = musaStreamWaitEvent(dependent_stream, event, 0);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaStreamWaitEvent failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    // Event can be destroyed after wait is queued
    err = musaEventDestroy(event);
    if (err != musaSuccess) {
      LOG(WARNING) << "CreateStreamDependency: musaEventDestroy failed: "
                   << musaGetErrorString(err);
    }

    return true;
  }

  port::Status AllocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event) {
      return port::Status(port::error::INTERNAL,
                          "Invalid event implementation");
    }
    if (!musa_event->Init()) {
      return port::Status(port::error::INTERNAL,
                          "Failed to initialize MUSA event");
    }
    return port::Status::OK();
  }

  port::Status DeallocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (musa_event && musa_event->handle()) {
      musaEventDestroy(musa_event->handle());
    }
    return port::Status::OK();
  }

  port::Status RecordEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return port::Status(port::error::INTERNAL, "Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaEventRecord(musa_event->handle(), mstream);
    if (err != musaSuccess) {
      return port::Status(port::error::INTERNAL, "musaEventRecord failed");
    }
    return port::Status::OK();
  }

  port::Status WaitForEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return port::Status(port::error::INTERNAL, "Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaStreamWaitEvent(mstream, musa_event->handle(), 0);
    if (err != musaSuccess) {
      return port::Status(port::error::INTERNAL, "musaStreamWaitEvent failed");
    }
    return port::Status::OK();
  }

  Event::Status PollForEventStatus(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return Event::Status::kError;
    }
    musaError_t err = musaEventQuery(musa_event->handle());
    if (err == musaSuccess) return Event::Status::kComplete;
    if (err == musaErrorNotReady) return Event::Status::kPending;
    return Event::Status::kError;
  }

 private:
  PluginConfig plugin_config_;
  int device_ordinal_;
};

}  // namespace musa
}  // namespace stream_executor

#else

#include <cstdint>
#include <memory>
#include <optional>

#include "tsl/platform/errors.h"

#include "absl/functional/any_invocable.h"

#include "musa_device.h"
#include "musa_event.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "musa_stream.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/stream_executor/stream_executor_internal.h"
namespace stream_executor {
namespace musa {

inline tsl::Status FromMusaStatus(mStatus s) {
  if (s == mStatus::SUCCESS) {
    return ::tsl::OkStatus();
  }
  return tsl::errors::Internal("MUSA Operation Failed");
}

class MusaExecutor : public internal::StreamExecutorInterface {
 public:
  MusaExecutor() = default;
  ~MusaExecutor() override {}

  tsl::Status Init(int device_ordinal, DeviceOptions device_options) override {
    device_ordinal_ = device_ordinal;
    return ::tsl::OkStatus();
  }

  void SetNextStream(musaStream_t stream) { next_stream_ = stream; }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    if (next_stream_ != nullptr) {
      musaStream_t h = next_stream_;
      next_stream_ = nullptr;
      return std::make_unique<MusaStream>(h);
    }

    musaStream_t h;
    musaError_t err = musaStreamCreate(&h);
    if (err != musaSuccess) {
      LOG(ERROR) << "musaStreamCreate failed: " << musaGetErrorString(err);
      return nullptr;
    }
    return std::make_unique<MusaStream>(h);
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return std::make_unique<MusaEvent>();
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }


  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override {
    if (size == 0) {
      return DeviceMemoryBase(nullptr, 0);
    }
    musaSetDevice(device_ordinal_);
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);
    if (err != musaSuccess) {
      LOG(ERROR) << "MusaExecutor::Allocate failed for " << size
                 << " bytes: " << musaGetErrorString(err);
      return DeviceMemoryBase(nullptr, 0);
    }
    return DeviceMemoryBase(ptr, size);
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                     uint64_t size) override {
    return reinterpret_cast<char*>(parent->opaque()) + offset;
  }

  void Deallocate(DeviceMemoryBase* mem) override {
    if (mem && mem->opaque()) {
      musaSetDevice(device_ordinal_);
      musaError_t err = musaFree(mem->opaque());
      if (err != musaSuccess) {
        LOG(ERROR) << "MUSA Deallocate failed: " << musaGetErrorString(err);
      }
    }
  }

  bool HostMemoryRegister(void* mem, uint64_t size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  void* HostMemoryAllocate(uint64_t size) override { return nullptr; }
  void HostMemoryDeallocate(void* mem) override {}

  tsl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override {
    mHandle h;

    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  tsl::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64_t size) override {
    mHandle h;
    return FromMusaStatus(tensorflow::musa::Memset(
        h, location->opaque(), size, static_cast<uint8_t>(value)));
  }

  tsl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override {
    // H2D
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyH2D(gpu_dst->opaque(), host_src, size));
  }

  tsl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override {
    // D2H
    return FromMusaStatus(
        tensorflow::musa::MusaMemcpyD2H(host_dst, gpu_src.opaque(), size));
  }

  tsl::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64_t size) override {
    // D2D
    return FromMusaStatus(tensorflow::musa::MusaMemcpyD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size));
  }

  musaStream_t GetMusaStream(Stream* stream) {
    auto* musa_stream_impl = static_cast<MusaStream*>(stream->implementation());

    return musa_stream_impl->GetStream();
  }

  // D2D Async
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2D(
        gpu_dst->opaque(), gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // H2D Async
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncH2D(
        gpu_dst->opaque(), host_src, size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // D2H Async
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override {
    auto status = tensorflow::musa::MusaMemcpyAsyncD2H(
        host_dst, gpu_src.opaque(), size, GetMusaStream(stream));
    return status == mStatus::SUCCESS;
  }

  // MemZero Async
  tsl::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  // Memset32 Async
  tsl::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32_t pattern, uint64_t size) override {
    mHandle h;
    h.SetStream(GetMusaStream(stream));
    return FromMusaStatus(
        tensorflow::musa::Memset32(h, location->opaque(), size, pattern));
  }

  tsl::Status BlockHostUntilDone(Stream* stream) override {
    internal::StreamInterface* implementation = stream->implementation();
    auto* musa_stream = static_cast<MusaStream*>(implementation);
    return musa_stream->BlockHostUntilDone_DEBUG(stream);
  }

  bool HostCallback(Stream* stream,
                    absl::AnyInvocable<tsl::Status() &&> callback) override {
    // Execute callback asynchronously via a host function
    // This ensures the callback runs after all preceding stream operations
    musaStream_t musa_stream = GetMusaStream(stream);
    musaError_t err = musaLaunchHostFunc(musa_stream,
        [](void* user_data) {
          auto* cb = static_cast<absl::AnyInvocable<tsl::Status() &&>*>(user_data);
          std::move(*cb)();
          delete cb;
        },
        new absl::AnyInvocable<tsl::Status() &&>(std::move(callback)));
    if (err != musaSuccess) {
      LOG(WARNING) << "MusaExecutor::HostCallback failed: "
                   << musaGetErrorString(err);
      return false;
    }
    return true;
  }

  tsl::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return ::tsl::OkStatus();
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return false;
  }

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    internal::DeviceDescriptionBuilder builder;
    builder.set_name("MUSA Device");
    return builder.Build();
  }

  bool SynchronizeAllActivity() override { return true; }
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override {
    return false;
  }
  bool AllocateStream(Stream* stream) override { return true; }
  void DeallocateStream(Stream* stream) override {}
  bool CreateStreamDependency(Stream* dependent, Stream* other) override {
    // Create an event on 'other' stream and wait on 'dependent' stream
    musaEvent_t event;
    musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventCreate failed: "
                 << musaGetErrorString(err);
      return false;
    }

    musaStream_t other_stream = GetMusaStream(other);
    err = musaEventRecord(event, other_stream);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaEventRecord failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    musaStream_t dependent_stream = GetMusaStream(dependent);
    err = musaStreamWaitEvent(dependent_stream, event, 0);
    if (err != musaSuccess) {
      LOG(ERROR) << "CreateStreamDependency: musaStreamWaitEvent failed: "
                 << musaGetErrorString(err);
      musaEventDestroy(event);
      return false;
    }

    // Event can be destroyed after wait is queued
    err = musaEventDestroy(event);
    if (err != musaSuccess) {
      LOG(WARNING) << "CreateStreamDependency: musaEventDestroy failed: "
                   << musaGetErrorString(err);
    }

    return true;
  }

  tsl::Status AllocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event) {
      return tsl::errors::Internal("Invalid event implementation");
    }
    if (!musa_event->Init()) {
      return tsl::errors::Internal("Failed to initialize MUSA event");
    }
    return ::tsl::OkStatus();
  }

  tsl::Status DeallocateEvent(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (musa_event && musa_event->handle()) {
      musaEventDestroy(musa_event->handle());
    }
    return ::tsl::OkStatus();
  }

  tsl::Status RecordEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return tsl::errors::Internal("Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaEventRecord(musa_event->handle(), mstream);
    if (err != musaSuccess) {
      return tsl::errors::Internal("musaEventRecord failed");
    }
    return ::tsl::OkStatus();
  }

  tsl::Status WaitForEvent(Stream* stream, Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return tsl::errors::Internal("Invalid event");
    }
    musaStream_t mstream = GetMusaStream(stream);
    musaError_t err = musaStreamWaitEvent(mstream, musa_event->handle(), 0);
    if (err != musaSuccess) {
      return tsl::errors::Internal("musaStreamWaitEvent failed");
    }
    return ::tsl::OkStatus();
  }

  Event::Status PollForEventStatus(Event* event) override {
    auto* musa_event = static_cast<MusaEvent*>(event->implementation());
    if (!musa_event || !musa_event->handle()) {
      return Event::Status::kError;
    }
    musaError_t err = musaEventQuery(musa_event->handle());
    if (err == musaSuccess) return Event::Status::kComplete;
    if (err == musaErrorNotReady) return Event::Status::kPending;
    return Event::Status::kError;
  }

 private:
  int device_ordinal_;
  musaStream_t next_stream_ = nullptr;
};

}  // namespace musa
}  // namespace stream_executor

#endif

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_
