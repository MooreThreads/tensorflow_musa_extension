/* Copyright 2025 The TensorFlow MUSA Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_TELEMETRY_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_TELEMETRY_H_

#include <musa_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

// Configuration for the telemetry system.
// Controlled via environment variables:
//   MUSA_TELEMETRY_ENABLED=1  - Enable telemetry (default: disabled)
//   MUSA_TELEMETRY_LOG_PATH   - Path to log file (default: stderr)
//   MUSA_TELEMETRY_BUFFER_SIZE - Event buffer size (default: 10000)
//   MUSA_TELEMETRY_FLUSH_MS   - Flush interval in ms (default: 100)
struct TelemetryConfig {
  bool enabled = false;
  std::string log_path;
  size_t buffer_size = 10000;
  int flush_interval_ms = 100;
  bool include_stack_trace = false;

  static TelemetryConfig FromEnv();
};

// Event types for structured logging.
enum class TelemetryEventType : int {
  kTensorAllocate = 0,
  kTensorFree = 1,
  kKernelLaunch = 2,
  kMemcpyH2D = 3,
  kMemcpyD2H = 4,
  kMemcpyD2D = 5,
  kEventRecord = 6,
  kEventWait = 7,
  kEventSync = 8,
  kStreamSync = 9,
  kDeviceSync = 10,
  kDirtyDataDetected = 11,
  kCustom = 12
};

// Convert TelemetryEventType to string for JSON output.
const char* TelemetryEventTypeToString(TelemetryEventType type);

// Structured event record for telemetry logging.
struct TelemetryEvent {
  // Timestamp in nanoseconds since epoch.
  uint64_t timestamp_ns;

  // Event type.
  TelemetryEventType event_type;

  // Unique correlation ID for request tracking.
  uint64_t correlation_id;

  // Device ID (GPU index).
  int device_id;

  // Stream ID (hash of musaStream_t pointer).
  uint64_t stream_id;

  // Memory address (for tensor/memory events).
  uintptr_t memory_addr;

  // Memory size in bytes.
  size_t memory_size;

  // Tensor ID (if applicable).
  uint64_t tensor_id;

  // Operation name (e.g., "MatMul", "Conv2D").
  std::string op_name;

  // Input tensor IDs (for kernel launches).
  std::vector<uint64_t> input_tensor_ids;

  // Output tensor IDs (for kernel launches).
  std::vector<uint64_t> output_tensor_ids;

  // Event handle (for MUSA event operations).
  uintptr_t event_handle;

  // Source stream ID (for event wait operations).
  uint64_t source_stream_id;

  // Additional metadata as key-value pairs.
  std::unordered_map<std::string, std::string> metadata;

  // Thread ID where the event was recorded.
  uint64_t thread_id;

  TelemetryEvent();
  std::string ToJson() const;
};

// Memory operation record for dirty data backtrace.
struct MemoryOpRecord {
  uint64_t timestamp_ns;
  TelemetryEventType op_type;
  uintptr_t memory_addr;
  size_t size;
  uint64_t tensor_id;
  uint64_t stream_id;
  std::string op_name;
  uint64_t correlation_id;
};

// Global telemetry manager.
// This is a singleton that manages the event buffer, logging thread,
// and provides the backtrace API for dirty data investigation.
class MusaTelemetry {
 public:
  // Get the singleton instance.
  static MusaTelemetry& Instance();

  // Initialize telemetry with configuration.
  // Should be called once at plugin load time.
  void Initialize(const TelemetryConfig& config);

  // Shutdown telemetry.
  // Flushes remaining events and stops the logging thread.
  void Shutdown();

  // Check if telemetry is enabled.
  bool IsEnabled() const { return config_.enabled && enabled_.load(); }

  // Record a telemetry event.
  // This is non-blocking and adds the event to an async queue.
  void RecordEvent(TelemetryEvent event);

  // Convenience methods for common event types.
  void OnTensorAllocate(uint64_t tensor_id, void* addr, size_t size,
                        int device_id, uint64_t stream_id);
  void OnTensorFree(uint64_t tensor_id, void* addr, size_t size, int device_id);
  void OnKernelLaunch(const std::string& op_name, int device_id,
                      uint64_t stream_id,
                      const std::vector<uint64_t>& input_tensor_ids,
                      const std::vector<uint64_t>& output_tensor_ids);
  void OnMemcpy(void* dst, void* src, size_t size, int device_id,
                uint64_t stream_id, TelemetryEventType type);
  void OnEventRecord(musaEvent_t event, uint64_t stream_id, int device_id);
  void OnEventWait(musaEvent_t event, uint64_t waiting_stream_id,
                   uint64_t source_stream_id, int device_id);
  void OnDirtyDataDetected(void* addr, size_t size, int device_id,
                           const std::string& description);

  // Backtrace API: Get the last N operations on a memory address.
  // Returns operations in reverse chronological order (most recent first).
  std::vector<MemoryOpRecord> BacktraceByAddress(void* addr, size_t count = 10);

  // Backtrace API: Get operations within a time range.
  std::vector<MemoryOpRecord> BacktraceByTime(uint64_t start_ns, uint64_t end_ns);

  // Backtrace API: Get operations related to a tensor ID.
  std::vector<MemoryOpRecord> BacktraceByTensorId(uint64_t tensor_id,
                                                   size_t count = 20);

  // Get system health snapshot for monitoring.
  std::string GetHealthSnapshot();

  // Generate correlation ID for request tracking.
  uint64_t NewCorrelationId();

  // Get the current tensor ID counter (for external use).
  uint64_t NewTensorId();

  // Convert stream pointer to stream ID.
  static uint64_t StreamToId(musaStream_t stream);

 private:
  MusaTelemetry();
  ~MusaTelemetry();

  // Prevent copy and move.
  MusaTelemetry(const MusaTelemetry&) = delete;
  MusaTelemetry& operator=(const MusaTelemetry&) = delete;

  // Logging thread function.
  void LoggingThread();

  // Flush events to output.
  void FlushEvents();

  // Update memory operation index for backtrace.
  void UpdateMemoryIndex(const TelemetryEvent& event);

  // Write event to output (file or stderr).
  void WriteEvent(const TelemetryEvent& event);

  TelemetryConfig config_;
  std::atomic<bool> enabled_{false};
  std::atomic<uint64_t> correlation_id_counter_{0};
  std::atomic<uint64_t> tensor_id_counter_{0};

  // Event buffer with mutex protection.
  mutex event_mutex_;
  std::deque<TelemetryEvent> event_buffer_ TF_GUARDED_BY(event_mutex_);
  condition_variable event_cv_;

  // Memory operation index for backtrace.
  // Maps memory address (page-aligned) to list of operations.
  mutex index_mutex_;
  static constexpr size_t kMemoryPageSize = 4096;
  std::unordered_map<uintptr_t, std::deque<MemoryOpRecord>> memory_index_
      TF_GUARDED_BY(index_mutex_);

  // Logging thread.
  std::thread logging_thread_;
  std::atomic<bool> shutdown_{false};

  // Output file stream (nullptr for stderr).
  FILE* output_file_ = nullptr;

  // Statistics.
  std::atomic<uint64_t> events_logged_{0};
  std::atomic<uint64_t> events_dropped_{0};
};

// RAII helper for timing scopes.
class TelemetryScope {
 public:
  TelemetryScope(const std::string& op_name, int device_id, uint64_t stream_id);
  ~TelemetryScope();

 private:
  std::string op_name_;
  int device_id_;
  uint64_t stream_id_;
  uint64_t start_ns_;
  uint64_t correlation_id_;
};

// Compile-time enabled/disabled macros.
#ifndef MUSA_DISABLE_TRACE_LOGGING

#define MUSA_TELEMETRY_INIT(config) \
  ::tensorflow::musa::MusaTelemetry::Instance().Initialize(config)

#define MUSA_TELEMETRY_SHUTDOWN() \
  ::tensorflow::musa::MusaTelemetry::Instance().Shutdown()

#define MUSA_TELEMETRY_ENABLED() \
  ::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()

#define MUSA_TELEMETRY_ON_TENSOR_ALLOCATE(tensor_id, addr, size, device_id, \
                                          stream_id)                        \
  do {                                                                       \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) {        \
      ::tensorflow::musa::MusaTelemetry::Instance().OnTensorAllocate(       \
          tensor_id, addr, size, device_id, stream_id);                     \
    }                                                                        \
  } while (0)

#define MUSA_TELEMETRY_ON_TENSOR_FREE(tensor_id, addr, size, device_id) \
  do {                                                                  \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) {   \
      ::tensorflow::musa::MusaTelemetry::Instance().OnTensorFree(      \
          tensor_id, addr, size, device_id);                            \
    }                                                                   \
  } while (0)

#define MUSA_TELEMETRY_ON_KERNEL_LAUNCH(op_name, device_id, stream_id, \
                                        inputs, outputs)              \
  do {                                                                \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) { \
      ::tensorflow::musa::MusaTelemetry::Instance().OnKernelLaunch(  \
          op_name, device_id, stream_id, inputs, outputs);           \
    }                                                                 \
  } while (0)

#define MUSA_TELEMETRY_ON_MEMCPY(dst, src, size, device_id, stream_id, type) \
  do {                                                                       \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) {        \
      ::tensorflow::musa::MusaTelemetry::Instance().OnMemcpy(               \
          dst, src, size, device_id, stream_id, type);                      \
    }                                                                        \
  } while (0)

#define MUSA_TELEMETRY_ON_EVENT_RECORD(event, stream_id, device_id) \
  do {                                                              \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) { \
      ::tensorflow::musa::MusaTelemetry::Instance().OnEventRecord(  \
          event, stream_id, device_id);                             \
    }                                                               \
  } while (0)

#define MUSA_TELEMETRY_ON_EVENT_WAIT(event, waiting_stream, source_stream, \
                                     device_id)                            \
  do {                                                                     \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) {      \
      ::tensorflow::musa::MusaTelemetry::Instance().OnEventWait(          \
          event, waiting_stream, source_stream, device_id);               \
    }                                                                      \
  } while (0)

#define MUSA_TELEMETRY_ON_DIRTY_DATA(addr, size, device_id, desc) \
  do {                                                            \
    if (::tensorflow::musa::MusaTelemetry::Instance().IsEnabled()) { \
      ::tensorflow::musa::MusaTelemetry::Instance().OnDirtyDataDetected( \
          addr, size, device_id, desc);                           \
    }                                                             \
  } while (0)

#define MUSA_TELEMETRY_BACKTRACE(addr, count) \
  ::tensorflow::musa::MusaTelemetry::Instance().BacktraceByAddress(addr, count)

#define MUSA_TELEMETRY_NEW_TENSOR_ID() \
  ::tensorflow::musa::MusaTelemetry::Instance().NewTensorId()

#define MUSA_TELEMETRY_STREAM_ID(stream) \
  ::tensorflow::musa::MusaTelemetry::StreamToId(stream)

#else  // MUSA_DISABLE_TRACE_LOGGING

#define MUSA_TELEMETRY_INIT(config) do {} while (0)
#define MUSA_TELEMETRY_SHUTDOWN() do {} while (0)
#define MUSA_TELEMETRY_ENABLED() false
#define MUSA_TELEMETRY_ON_TENSOR_ALLOCATE(tensor_id, addr, size, device_id, stream_id) do {} while (0)
#define MUSA_TELEMETRY_ON_TENSOR_FREE(tensor_id, addr, size, device_id) do {} while (0)
#define MUSA_TELEMETRY_ON_KERNEL_LAUNCH(op_name, device_id, stream_id, inputs, outputs) do {} while (0)
#define MUSA_TELEMETRY_ON_MEMCPY(dst, src, size, device_id, stream_id, type) do {} while (0)
#define MUSA_TELEMETRY_ON_EVENT_RECORD(event, stream_id, device_id) do {} while (0)
#define MUSA_TELEMETRY_ON_EVENT_WAIT(event, waiting_stream, source_stream, device_id) do {} while (0)
#define MUSA_TELEMETRY_ON_DIRTY_DATA(addr, size, device_id, desc) do {} while (0)
#define MUSA_TELEMETRY_BACKTRACE(addr, count) std::vector<::tensorflow::musa::MemoryOpRecord>()
#define MUSA_TELEMETRY_NEW_TENSOR_ID() 0
#define MUSA_TELEMETRY_STREAM_ID(stream) 0

#endif  // MUSA_DISABLE_TRACE_LOGGING

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_TELEMETRY_H_