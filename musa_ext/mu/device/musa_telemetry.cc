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

#include "mu/device/musa_telemetry.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

namespace tensorflow {
namespace musa {

// Helper to get current timestamp in nanoseconds.
static uint64_t NowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Helper to get thread ID.
static uint64_t GetThreadId() {
  std::hash<std::thread::id> hasher;
  return hasher(std::this_thread::get_id());
}

// Helper to escape string for JSON.
static std::string JsonEscape(const std::string& s) {
  std::string result;
  result.reserve(s.size() * 2);
  for (char c : s) {
    switch (c) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned int>(c));
          result += buf;
        } else {
          result += c;
        }
    }
  }
  return result;
}

// TelemetryConfig implementation.

TelemetryConfig TelemetryConfig::FromEnv() {
  TelemetryConfig config;

  const char* enabled_env = std::getenv("MUSA_TELEMETRY_ENABLED");
  if (enabled_env != nullptr && enabled_env[0] != '\0') {
    config.enabled = (std::string(enabled_env) == "1" ||
                      std::string(enabled_env) == "true");
  }

  const char* log_path_env = std::getenv("MUSA_TELEMETRY_LOG_PATH");
  if (log_path_env != nullptr && log_path_env[0] != '\0') {
    config.log_path = log_path_env;
  }

  const char* buffer_size_env = std::getenv("MUSA_TELEMETRY_BUFFER_SIZE");
  if (buffer_size_env != nullptr && buffer_size_env[0] != '\0') {
    config.buffer_size = static_cast<size_t>(std::stoull(buffer_size_env));
  }

  const char* flush_env = std::getenv("MUSA_TELEMETRY_FLUSH_MS");
  if (flush_env != nullptr && flush_env[0] != '\0') {
    config.flush_interval_ms = std::stoi(flush_env);
  }

  const char* stack_env = std::getenv("MUSA_TELEMETRY_STACK_TRACE");
  if (stack_env != nullptr && stack_env[0] != '\0') {
    config.include_stack_trace = (std::string(stack_env) == "1" ||
                                  std::string(stack_env) == "true");
  }

  return config;
}

// TelemetryEventType to string.
const char* TelemetryEventTypeToString(TelemetryEventType type) {
  switch (type) {
    case TelemetryEventType::kTensorAllocate:
      return "tensor_allocate";
    case TelemetryEventType::kTensorFree:
      return "tensor_free";
    case TelemetryEventType::kKernelLaunch:
      return "kernel_launch";
    case TelemetryEventType::kMemcpyH2D:
      return "memcpy_h2d";
    case TelemetryEventType::kMemcpyD2H:
      return "memcpy_d2h";
    case TelemetryEventType::kMemcpyD2D:
      return "memcpy_d2d";
    case TelemetryEventType::kEventRecord:
      return "event_record";
    case TelemetryEventType::kEventWait:
      return "event_wait";
    case TelemetryEventType::kEventSync:
      return "event_sync";
    case TelemetryEventType::kStreamSync:
      return "stream_sync";
    case TelemetryEventType::kDeviceSync:
      return "device_sync";
    case TelemetryEventType::kDirtyDataDetected:
      return "dirty_data_detected";
    case TelemetryEventType::kCustom:
      return "custom";
    default:
      return "unknown";
  }
}

// TelemetryEvent implementation.

TelemetryEvent::TelemetryEvent()
    : timestamp_ns(0),
      event_type(TelemetryEventType::kCustom),
      correlation_id(0),
      device_id(-1),
      stream_id(0),
      memory_addr(0),
      memory_size(0),
      tensor_id(0),
      event_handle(0),
      source_stream_id(0),
      thread_id(GetThreadId()) {}

std::string TelemetryEvent::ToJson() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(0);

  oss << "{\"timestamp_ns\":" << timestamp_ns;
  oss << ",\"event_type\":\"" << TelemetryEventTypeToString(event_type) << "\"";
  oss << ",\"correlation_id\":" << correlation_id;
  oss << ",\"device_id\":" << device_id;
  oss << ",\"stream_id\":" << stream_id;
  oss << ",\"thread_id\":" << thread_id;

  if (memory_addr != 0) {
    oss << ",\"memory_addr\":\"0x" << std::hex << memory_addr << std::dec << "\"";
  }
  if (memory_size != 0) {
    oss << ",\"memory_size\":" << memory_size;
  }
  if (tensor_id != 0) {
    oss << ",\"tensor_id\":" << tensor_id;
  }
  if (!op_name.empty()) {
    oss << ",\"op_name\":\"" << JsonEscape(op_name) << "\"";
  }
  if (!input_tensor_ids.empty()) {
    oss << ",\"input_tensor_ids\":[";
    for (size_t i = 0; i < input_tensor_ids.size(); ++i) {
      if (i > 0) oss << ",";
      oss << input_tensor_ids[i];
    }
    oss << "]";
  }
  if (!output_tensor_ids.empty()) {
    oss << ",\"output_tensor_ids\":[";
    for (size_t i = 0; i < output_tensor_ids.size(); ++i) {
      if (i > 0) oss << ",";
      oss << output_tensor_ids[i];
    }
    oss << "]";
  }
  if (event_handle != 0) {
    oss << ",\"event_handle\":\"0x" << std::hex << event_handle << std::dec
        << "\"";
  }
  if (source_stream_id != 0) {
    oss << ",\"source_stream_id\":" << source_stream_id;
  }
  if (!metadata.empty()) {
    oss << ",\"metadata\":{";
    bool first = true;
    for (const auto& kv : metadata) {
      if (!first) oss << ",";
      oss << "\"" << JsonEscape(kv.first) << "\":\"" << JsonEscape(kv.second)
          << "\"";
      first = false;
    }
    oss << "}";
  }

  oss << "}";
  return oss.str();
}

// MusaTelemetry implementation.

MusaTelemetry& MusaTelemetry::Instance() {
  static MusaTelemetry instance;
  return instance;
}

MusaTelemetry::MusaTelemetry() : output_file_(nullptr) {}

MusaTelemetry::~MusaTelemetry() { Shutdown(); }

void MusaTelemetry::Initialize(const TelemetryConfig& config) {
  config_ = config;

  if (!config_.enabled) {
    return;
  }

  // Open output file if specified.
  if (!config_.log_path.empty()) {
    output_file_ = std::fopen(config_.log_path.c_str(), "a");
    if (output_file_ == nullptr) {
      LOG(WARNING) << "[MUSA Telemetry] Failed to open log file: "
                   << config_.log_path << ", falling back to stderr";
    }
  }

  // Start logging thread.
  shutdown_.store(false);
  enabled_.store(true);
  logging_thread_ = std::thread(&MusaTelemetry::LoggingThread, this);

  LOG(INFO) << "[MUSA Telemetry] Initialized with buffer_size="
            << config_.buffer_size << ", flush_interval_ms="
            << config_.flush_interval_ms;
}

void MusaTelemetry::Shutdown() {
  if (!enabled_.load()) {
    return;
  }

  enabled_.store(false);
  shutdown_.store(true);

  // Wake up logging thread.
  {
    mutex_lock l(event_mutex_);
    event_cv_.notify_all();
  }

  // Wait for logging thread to finish.
  if (logging_thread_.joinable()) {
    logging_thread_.join();
  }

  // Close output file.
  if (output_file_ != nullptr) {
    std::fclose(output_file_);
    output_file_ = nullptr;
  }

  LOG(INFO) << "[MUSA Telemetry] Shutdown. Events logged: " << events_logged_
            << ", Events dropped: " << events_dropped_;
}

void MusaTelemetry::RecordEvent(TelemetryEvent event) {
  if (!IsEnabled()) {
    return;
  }

  event.timestamp_ns = NowNs();
  event.correlation_id = correlation_id_counter_.fetch_add(1);

  {
    mutex_lock l(event_mutex_);

    // Drop oldest event if buffer is full.
    if (event_buffer_.size() >= config_.buffer_size) {
      event_buffer_.pop_front();
      events_dropped_.fetch_add(1);
    }

    event_buffer_.push_back(event);
    event_cv_.notify_one();
  }

  // Update memory index for backtrace.
  UpdateMemoryIndex(event);
}

void MusaTelemetry::OnTensorAllocate(uint64_t tensor_id, void* addr,
                                     size_t size, int device_id,
                                     uint64_t stream_id) {
  TelemetryEvent event;
  event.event_type = TelemetryEventType::kTensorAllocate;
  event.tensor_id = tensor_id;
  event.memory_addr = reinterpret_cast<uintptr_t>(addr);
  event.memory_size = size;
  event.device_id = device_id;
  event.stream_id = stream_id;
  event.op_name = "Allocate";
  RecordEvent(event);
}

void MusaTelemetry::OnTensorFree(uint64_t tensor_id, void* addr, size_t size,
                                 int device_id) {
  TelemetryEvent event;
  event.event_type = TelemetryEventType::kTensorFree;
  event.tensor_id = tensor_id;
  event.memory_addr = reinterpret_cast<uintptr_t>(addr);
  event.memory_size = size;
  event.device_id = device_id;
  event.op_name = "Free";
  RecordEvent(event);
}

void MusaTelemetry::OnKernelLaunch(const std::string& op_name, int device_id,
                                   uint64_t stream_id,
                                   const std::vector<uint64_t>& input_tensor_ids,
                                   const std::vector<uint64_t>& output_tensor_ids) {
  TelemetryEvent event;
  event.event_type = TelemetryEventType::kKernelLaunch;
  event.op_name = op_name;
  event.device_id = device_id;
  event.stream_id = stream_id;
  event.input_tensor_ids = input_tensor_ids;
  event.output_tensor_ids = output_tensor_ids;
  RecordEvent(event);
}

void MusaTelemetry::OnMemcpy(void* dst, void* src, size_t size, int device_id,
                             uint64_t stream_id, TelemetryEventType type) {
  TelemetryEvent event;
  event.event_type = type;
  event.memory_addr = reinterpret_cast<uintptr_t>(dst);
  event.memory_size = size;
  event.device_id = device_id;
  event.stream_id = stream_id;

  // Store source address in metadata.
  std::ostringstream oss;
  oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(src);
  event.metadata["src_addr"] = oss.str();

  if (type == TelemetryEventType::kMemcpyH2D) {
    event.op_name = "MemcpyH2D";
  } else if (type == TelemetryEventType::kMemcpyD2H) {
    event.op_name = "MemcpyD2H";
  } else {
    event.op_name = "MemcpyD2D";
  }

  RecordEvent(event);
}

void MusaTelemetry::OnEventRecord(musaEvent_t event, uint64_t stream_id,
                                  int device_id) {
  TelemetryEvent telemetry_event;
  telemetry_event.event_type = TelemetryEventType::kEventRecord;
  telemetry_event.event_handle = reinterpret_cast<uintptr_t>(event);
  telemetry_event.stream_id = stream_id;
  telemetry_event.device_id = device_id;
  telemetry_event.op_name = "EventRecord";
  RecordEvent(telemetry_event);
}

void MusaTelemetry::OnEventWait(musaEvent_t event, uint64_t waiting_stream_id,
                                uint64_t source_stream_id, int device_id) {
  TelemetryEvent telemetry_event;
  telemetry_event.event_type = TelemetryEventType::kEventWait;
  telemetry_event.event_handle = reinterpret_cast<uintptr_t>(event);
  telemetry_event.stream_id = waiting_stream_id;
  telemetry_event.source_stream_id = source_stream_id;
  telemetry_event.device_id = device_id;
  telemetry_event.op_name = "EventWait";
  RecordEvent(telemetry_event);
}

void MusaTelemetry::OnDirtyDataDetected(void* addr, size_t size, int device_id,
                                        const std::string& description) {
  TelemetryEvent event;
  event.event_type = TelemetryEventType::kDirtyDataDetected;
  event.memory_addr = reinterpret_cast<uintptr_t>(addr);
  event.memory_size = size;
  event.device_id = device_id;
  event.op_name = "DirtyDataDetected";
  event.metadata["description"] = description;
  RecordEvent(event);

  // Immediately flush to ensure we capture this event.
  FlushEvents();

  LOG(ERROR) << "[MUSA Telemetry] DIRTY DATA DETECTED at address " << addr
             << ", size=" << size << ", device_id=" << device_id
             << ", description=" << description;
}

std::vector<MemoryOpRecord> MusaTelemetry::BacktraceByAddress(void* addr,
                                                              size_t count) {
  std::vector<MemoryOpRecord> result;
  if (!IsEnabled()) {
    return result;
  }

  uintptr_t aligned_addr =
      reinterpret_cast<uintptr_t>(addr) & ~(kMemoryPageSize - 1);

  mutex_lock l(index_mutex_);
  auto it = memory_index_.find(aligned_addr);
  if (it != memory_index_.end()) {
    const auto& records = it->second;
    size_t n = std::min(count, records.size());
    result.reserve(n);
    // Return in reverse order (most recent first).
    auto rit = records.rbegin();
    for (size_t i = 0; i < n && rit != records.rend(); ++i, ++rit) {
      result.push_back(*rit);
    }
  }

  return result;
}

std::vector<MemoryOpRecord> MusaTelemetry::BacktraceByTime(uint64_t start_ns,
                                                           uint64_t end_ns) {
  std::vector<MemoryOpRecord> result;
  if (!IsEnabled()) {
    return result;
  }

  mutex_lock l(index_mutex_);
  for (const auto& kv : memory_index_) {
    for (const auto& record : kv.second) {
      if (record.timestamp_ns >= start_ns && record.timestamp_ns <= end_ns) {
        result.push_back(record);
      }
    }
  }

  // Sort by timestamp (most recent first).
  std::sort(result.begin(), result.end(),
            [](const MemoryOpRecord& a, const MemoryOpRecord& b) {
              return a.timestamp_ns > b.timestamp_ns;
            });

  return result;
}

std::vector<MemoryOpRecord> MusaTelemetry::BacktraceByTensorId(
    uint64_t tensor_id, size_t count) {
  std::vector<MemoryOpRecord> result;
  if (!IsEnabled()) {
    return result;
  }

  mutex_lock l(index_mutex_);
  for (const auto& kv : memory_index_) {
    for (const auto& record : kv.second) {
      if (record.tensor_id == tensor_id) {
        result.push_back(record);
      }
    }
  }

  // Sort by timestamp (most recent first).
  std::sort(result.begin(), result.end(),
            [](const MemoryOpRecord& a, const MemoryOpRecord& b) {
              return a.timestamp_ns > b.timestamp_ns;
            });

  if (result.size() > count) {
    result.resize(count);
  }

  return result;
}

std::string MusaTelemetry::GetHealthSnapshot() {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);

  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);

  oss << "{\n";
  oss << "  \"snapshot_time\": \""
      << std::put_time(std::localtime(&now_time), "%Y-%m-%dT%H:%M:%S") << "\",\n";
  oss << "  \"telemetry\": {\n";
  oss << "    \"enabled\": " << (enabled_.load() ? "true" : "false") << ",\n";
  oss << "    \"events_logged\": " << events_logged_.load() << ",\n";
  oss << "    \"events_dropped\": " << events_dropped_.load() << ",\n";

  {
    mutex_lock l(event_mutex_);
    oss << "    \"buffer_size\": " << event_buffer_.size() << ",\n";
    oss << "    \"buffer_capacity\": " << config_.buffer_size << ",\n";
  }

  {
    mutex_lock l(index_mutex_);
    oss << "    \"memory_pages_tracked\": " << memory_index_.size() << "\n";
  }

  oss << "  }\n";
  oss << "}\n";

  return oss.str();
}

uint64_t MusaTelemetry::NewCorrelationId() {
  return correlation_id_counter_.fetch_add(1);
}

uint64_t MusaTelemetry::NewTensorId() {
  return tensor_id_counter_.fetch_add(1);
}

uint64_t MusaTelemetry::StreamToId(musaStream_t stream) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
}

void MusaTelemetry::LoggingThread() {
  while (!shutdown_.load()) {
    FlushEvents();

    // Wait for new events or shutdown.
    {
      mutex_lock l(event_mutex_);
      // Use wait_for with timeout (no predicate) - check condition after wake
      event_cv_.wait_for(l, std::chrono::milliseconds(config_.flush_interval_ms));
      // Check for shutdown or pending events after waking
      if (shutdown_.load()) {
        break;
      }
    }
  }

  // Final flush on shutdown.
  FlushEvents();
}

void MusaTelemetry::FlushEvents() {
  std::vector<TelemetryEvent> to_flush;

  {
    mutex_lock l(event_mutex_);
    if (event_buffer_.empty()) {
      return;
    }
    to_flush.reserve(event_buffer_.size());
    while (!event_buffer_.empty()) {
      to_flush.push_back(std::move(event_buffer_.front()));
      event_buffer_.pop_front();
    }
  }

  // Write events outside the lock.
  for (const auto& event : to_flush) {
    WriteEvent(event);
    events_logged_.fetch_add(1);
  }
}

void MusaTelemetry::UpdateMemoryIndex(const TelemetryEvent& event) {
  // Only index memory-related events.
  if (event.event_type != TelemetryEventType::kTensorAllocate &&
      event.event_type != TelemetryEventType::kTensorFree &&
      event.event_type != TelemetryEventType::kKernelLaunch &&
      event.event_type != TelemetryEventType::kMemcpyH2D &&
      event.event_type != TelemetryEventType::kMemcpyD2H &&
      event.event_type != TelemetryEventType::kMemcpyD2D) {
    return;
  }

  if (event.memory_addr == 0) {
    return;
  }

  // Align to page boundary.
  uintptr_t aligned_addr = event.memory_addr & ~(kMemoryPageSize - 1);

  MemoryOpRecord record;
  record.timestamp_ns = event.timestamp_ns;
  record.op_type = event.event_type;
  record.memory_addr = event.memory_addr;
  record.size = event.memory_size;
  record.tensor_id = event.tensor_id;
  record.stream_id = event.stream_id;
  record.op_name = event.op_name;
  record.correlation_id = event.correlation_id;

  mutex_lock l(index_mutex_);
  auto& records = memory_index_[aligned_addr];
  records.push_back(record);

  // Keep last 100 operations per page to limit memory usage.
  while (records.size() > 100) {
    records.pop_front();
  }
}

void MusaTelemetry::WriteEvent(const TelemetryEvent& event) {
  std::string json = event.ToJson();
  json += "\n";

  if (output_file_ != nullptr) {
    std::fwrite(json.c_str(), 1, json.size(), output_file_);
    std::fflush(output_file_);
  } else {
    // Write to stderr in JSON Lines format.
    std::fprintf(stderr, "[MUSA_TELEMETRY] %s", json.c_str());
  }
}

// TelemetryScope implementation.

TelemetryScope::TelemetryScope(const std::string& op_name, int device_id,
                               uint64_t stream_id)
    : op_name_(op_name),
      device_id_(device_id),
      stream_id_(stream_id),
      start_ns_(NowNs()),
      correlation_id_(MusaTelemetry::Instance().NewCorrelationId()) {}

TelemetryScope::~TelemetryScope() {
  if (MusaTelemetry::Instance().IsEnabled()) {
    uint64_t end_ns = NowNs();
    uint64_t duration_ns = end_ns - start_ns_;

    TelemetryEvent event;
    event.event_type = TelemetryEventType::kCustom;
    event.op_name = op_name_ + "_duration";
    event.device_id = device_id_;
    event.stream_id = stream_id_;
    event.correlation_id = correlation_id_;
    event.metadata["duration_ns"] = std::to_string(duration_ns);
    MusaTelemetry::Instance().RecordEvent(event);
  }
}

}  // namespace musa
}  // namespace tensorflow