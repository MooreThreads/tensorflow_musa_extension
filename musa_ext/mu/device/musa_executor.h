#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_EXECUTOR_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include <memory>

#include "musa_stream.h" 
#include "musa_event.h"  
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "musa_device.h"
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
  explicit MusaExecutor(const PluginConfig& plugin_config) : plugin_config_(plugin_config) {}
  ~MusaExecutor() override {}

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
    device_ordinal_ = device_ordinal;
    return port::Status::OK();
  }

  // ========================================================================
  // 1. 工厂接口
  // ========================================================================
  
  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override {
      musaStream_t h;
      musaStreamCreate(&h);
      return std::make_unique<MusaStream>(h);
  }

  std::unique_ptr<internal::EventInterface> CreateEventImplementation() override {
    return std::make_unique<MusaEvent>();
  }

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation() override {
      return nullptr; 
  }

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
      return nullptr; 
  }

  // ========================================================================
  // 2. 内存管理接口
  // ========================================================================

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override { 
      
      return DeviceMemoryBase(nullptr, 0); 
  }
  
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset, uint64 size) override { 
      return reinterpret_cast<char*>(parent->opaque()) + offset;
  }

  void Deallocate(DeviceMemoryBase* mem) override {
      // musaFree(mem->opaque()); 
  }

  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  void* HostMemoryAllocate(uint64 size) override { return nullptr; }
  void HostMemoryDeallocate(void* mem) override {}

  // ========================================================================
  // 3. 内存拷贝 (同步)
  // ========================================================================
  
  port::Status SynchronousMemZero(DeviceMemoryBase* location, uint64 size) override { 
      // Memset 同步版，这里临时创建一个 handle
      mHandle h; 
      
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value, uint64 size) override { 
      mHandle h;
      
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, static_cast<uint8_t>(value)));
  }

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst, const void* host_src, uint64 size) override { 
      // H2D
      return FromMusaStatus(tensorflow::musa::MusaMemcpyH2D(gpu_dst->opaque(), host_src, size));
  }

  port::Status SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      // D2H
      return FromMusaStatus(tensorflow::musa::MusaMemcpyD2H(host_dst, gpu_src.opaque(), size));
  }

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      // D2D
      return FromMusaStatus(tensorflow::musa::MusaMemcpyD2D(gpu_dst->opaque(), gpu_src.opaque(), size));
  }
  
  // ========================================================================
  // 4. 内存拷贝 (异步)
  // ========================================================================

  // 从 TF Stream 中获取底层 musaStream_t
  musaStream_t GetMusaStream(Stream* stream) {
      auto* musa_stream_impl = static_cast<MusaStream*>(stream->implementation());
      
      return musa_stream_impl->GetStream(); 
  }

  // D2D Async
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncD2D(
          gpu_dst->opaque(), gpu_src.opaque(), size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // H2D Async
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncH2D(
          gpu_dst->opaque(), host_src, size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // D2H Async
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src, uint64 size) override { 
      auto status = tensorflow::musa::MusaMemcpyAsyncD2H(
          host_dst, gpu_src.opaque(), size, GetMusaStream(stream));
      return status == mStatus::SUCCESS;
  }

  // MemZero Async
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location, uint64 size) override { 
      mHandle h;
      h.SetStream(GetMusaStream(stream)); // 绑定流
      return FromMusaStatus(tensorflow::musa::Memset(h, location->opaque(), size, 0));
  }

  // Memset32 Async
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location, uint32 pattern, uint64 size) override { 
      mHandle h;
      h.SetStream(GetMusaStream(stream)); // 绑定流
      return FromMusaStatus(tensorflow::musa::Memset32(h, location->opaque(), size, pattern));
  }
  
  // ========================================================================
  // 5. 其他接口
  // ========================================================================

 
  port::Status BlockHostUntilDone(Stream* stream) override {
      internal::StreamInterface* implementation = stream->implementation();
      auto* musa_stream = static_cast<MusaStream*>(implementation);
      return musa_stream->BlockHostUntilDone_DEBUG(stream);
  }

  bool HostCallback(Stream* stream, std::function<port::Status()> callback) override { 

      return true; 
  }
  
  bool AllocateTimer(Timer* timer) override { return true; }
  void DeallocateTimer(Timer* timer) override {}
  bool StartTimer(Stream* stream, Timer* timer) override { return true; }
  bool StopTimer(Stream* stream, Timer* timer) override { return true; }

  int PlatformDeviceCount() override { return 1; }
  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override { return port::Status::OK(); }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override { return false; }

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription() const override {
      internal::DeviceDescriptionBuilder builder;
      builder.set_name("MUSA Device");
      return builder.Build();
  }

  bool SynchronizeAllActivity() override { return true; }
  bool DeviceMemoryUsage(int64* free, int64* total) const override { return false; }
  bool AllocateStream(Stream* stream) override { return true; }
  void DeallocateStream(Stream* stream) override {}
  bool CreateStreamDependency(Stream* dependent, Stream* other) override { return true; }

  port::Status AllocateEvent(Event* event) override { return port::Status::OK(); }
  port::Status DeallocateEvent(Event* event) override { return port::Status::OK(); }
  port::Status RecordEvent(Stream* stream, Event* event) override { return port::Status::OK(); }
  port::Status WaitForEvent(Stream* stream, Event* event) override { return port::Status::OK(); }
  Event::Status PollForEventStatus(Event* event) override { return Event::Status::kComplete; }

 private:
  PluginConfig plugin_config_;
  int device_ordinal_;
};

} // namespace musa
} // namespace stream_executor

#endif
