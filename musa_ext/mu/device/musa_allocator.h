#ifndef TENSORFLOW_MUSA_ALLOCATOR_H_
#define TENSORFLOW_MUSA_ALLOCATOR_H_

#include <musa_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mu/device/musa_telemetry.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

// Memory Coloring Configuration
// Enable with -DTF_MUSA_MEMORY_COLORING=1 compile flag or set at runtime
#ifdef TF_MUSA_MEMORY_COLORING
#define TF_MUSA_MEMORY_COLORING_ENABLED 1
#else
#define TF_MUSA_MEMORY_COLORING_ENABLED 0
#endif

// Magic numbers for memory coloring
// These patterns help detect uninitialized memory and Use-After-Free
// Using byte patterns since musaMemset works with bytes
constexpr uint8_t kMusaAllocByte = 0xAB;  // Filled on allocation (0xABABABAB pattern)
constexpr uint8_t kMusaFreeByte = 0xCD;   // Filled on free (0xCDCDCDCD pattern)
// For verification we check for these patterns
constexpr uint32_t kMusaAllocMagic = 0xABABABAB;  // Expected after alloc
constexpr uint32_t kMusaFreeMagic = 0xCDCDCDCD;   // Expected after free

// Memory coloring control - can be set at runtime
class MemoryColoringConfig {
 public:
  static MemoryColoringConfig& Instance() {
    static MemoryColoringConfig instance;
    return instance;
  }

  bool enabled() const { return enabled_.load(std::memory_order_acquire); }
  void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }

  bool track_history() const { return track_history_.load(std::memory_order_acquire); }
  void set_track_history(bool track) { track_history_.store(track, std::memory_order_release); }

  bool verify_on_free() const { return verify_on_free_.load(std::memory_order_acquire); }
  void set_verify_on_free(bool verify) { verify_on_free_.store(verify, std::memory_order_release); }

  // Size of guard bytes at start and end of each allocation (in uint32_t units)
  size_t guard_size() const { return 4; }  // 16 bytes

 private:
  MemoryColoringConfig()
      : enabled_(TF_MUSA_MEMORY_COLORING_ENABLED),
        track_history_(TF_MUSA_MEMORY_COLORING_ENABLED),
        verify_on_free_(TF_MUSA_MEMORY_COLORING_ENABLED) {}

  std::atomic<bool> enabled_{false};
  std::atomic<bool> track_history_{false};
  std::atomic<bool> verify_on_free_{false};
};

// Allocation record for tracking memory usage history
struct AllocationRecord {
  void* ptr;
  size_t size;
  size_t aligned_size;
  int device_id;
  uint64_t timestamp_ms;  // milliseconds since epoch
  uint64_t alloc_id;      // unique allocation ID
  uint32_t stream_id;     // MUSA stream hash (if available)
  bool active;            // true if allocated, false if freed
  std::string callstack;  // simplified callstack info

  // For freed allocations, track when and by whom
  uint64_t free_timestamp_ms;
  uint64_t free_alloc_id;  // ID of allocation that reused this address
};

// Memory tracking statistics
struct MemoryStats {
  uint64_t total_allocations{0};
  uint64_t total_frees{0};
  uint64_t active_allocations{0};
  uint64_t bytes_allocated{0};
  uint64_t bytes_freed{0};
  uint64_t bytes_active{0};
  uint64_t uaf_detected{0};
  uint64_t corruption_detected{0};
  uint64_t magic_mismatch{0};

  void Reset() {
    total_allocations = 0;
    total_frees = 0;
    active_allocations = 0;
    bytes_allocated = 0;
    bytes_freed = 0;
    bytes_active = 0;
    uaf_detected = 0;
    corruption_detected = 0;
    magic_mismatch = 0;
  }
};

// Global memory forensics tracker for allocation history and UAF detection
class MemoryForensicsTracker {
 public:
  static MemoryForensicsTracker& Instance() {
    static MemoryForensicsTracker instance;
    return instance;
  }

  // Record a new allocation
  uint64_t RecordAllocation(void* ptr, size_t size, int device_id,
                            uint32_t stream_id = 0) {
    if (!MemoryColoringConfig::Instance().track_history()) {
      return 0;
    }

    mutex_lock lock(mu_);
    uint64_t alloc_id = next_alloc_id_++;
    uint64_t timestamp = GetCurrentTimestampMs();

    AllocationRecord record;
    record.ptr = ptr;
    record.size = size;
    record.aligned_size = size;
    record.device_id = device_id;
    record.timestamp_ms = timestamp;
    record.alloc_id = alloc_id;
    record.stream_id = stream_id;
    record.active = true;

    // Store in history
    allocation_history_[ptr].push_back(record);
    active_allocations_[ptr] = record;

    // Update stats
    stats_.total_allocations++;
    stats_.active_allocations++;
    stats_.bytes_allocated += size;
    stats_.bytes_active += size;

    return alloc_id;
  }

  // Record a free operation and detect potential UAF
  void RecordFree(void* ptr, size_t size, int device_id) {
    if (!MemoryColoringConfig::Instance().track_history()) {
      return;
    }

    mutex_lock lock(mu_);
    uint64_t timestamp = GetCurrentTimestampMs();

    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      // Potential double-free or UAF
      LOG(WARNING) << "[MemoryForensics] Free of untracked address: " << ptr
                   << " size=" << size << " device=" << device_id;
      stats_.uaf_detected++;
      return;
    }

    AllocationRecord& record = it->second;
    record.active = false;
    record.free_timestamp_ms = timestamp;

    // Update history
    auto hist_it = allocation_history_.find(ptr);
    if (hist_it != allocation_history_.end()) {
      for (auto& hist_record : hist_it->second) {
        if (hist_record.alloc_id == record.alloc_id) {
          hist_record.active = false;
          hist_record.free_timestamp_ms = timestamp;
          break;
        }
      }
    }

    active_allocations_.erase(it);

    // Update stats
    stats_.total_frees++;
    stats_.active_allocations--;
    stats_.bytes_freed += size;
    stats_.bytes_active -= size;
  }

  // Check if address is currently allocated (for UAF detection)
  bool IsAddressAllocated(void* ptr) const {
    mutex_lock lock(mu_);
    return active_allocations_.find(ptr) != active_allocations_.end();
  }

  // Get allocation history for an address
  std::vector<AllocationRecord> GetAllocationHistory(void* ptr) const {
    mutex_lock lock(mu_);
    auto it = allocation_history_.find(ptr);
    if (it != allocation_history_.end()) {
      return it->second;
    }
    return {};
  }

  // Get current statistics
  MemoryStats GetStats() const {
    mutex_lock lock(mu_);
    return stats_;
  }

  // Reset statistics
  void ResetStats() {
    mutex_lock lock(mu_);
    stats_.Reset();
  }

  // Record corruption detection
  void RecordCorruption(const char* type) {
    mutex_lock lock(mu_);
    stats_.corruption_detected++;
    LOG(ERROR) << "[MemoryForensics] Corruption detected: " << type;
  }

  // Record magic number mismatch
  void RecordMagicMismatch(void* ptr, uint32_t expected, uint32_t actual) {
    mutex_lock lock(mu_);
    stats_.magic_mismatch++;
    LOG(ERROR) << "[MemoryForensics] Magic mismatch at " << ptr
               << " expected=0x" << std::hex << expected
               << " actual=0x" << actual << std::dec;
  }

  // Generate forensics report for debugging
  std::string GenerateReport(void* ptr) const {
    mutex_lock lock(mu_);
    std::ostringstream oss;
    oss << "# Memory Forensics Report\n";
    oss << "**Address:** " << ptr << "\n\n";

    auto hist_it = allocation_history_.find(ptr);
    if (hist_it != allocation_history_.end()) {
      oss << "## Allocation History\n";
      oss << "| Time (ms) | Operation | Size | Device | Active |\n";
      oss << "|-----------|-----------|------|--------|--------|\n";
      for (const auto& record : hist_it->second) {
        oss << "| " << record.timestamp_ms << " | ";
        oss << (record.active ? "ALLOC" : "FREE") << " | ";
        oss << record.size << " | " << record.device_id << " | ";
        oss << (record.active ? "YES" : "NO") << " |\n";
      }
    } else {
      oss << "No allocation history found for this address.\n";
    }

    oss << "\n## Statistics\n";
    oss << "- Total Allocations: " << stats_.total_allocations << "\n";
    oss << "- Active Allocations: " << stats_.active_allocations << "\n";
    oss << "- Bytes Active: " << stats_.bytes_active << "\n";
    oss << "- UAF Detected: " << stats_.uaf_detected << "\n";
    oss << "- Corruption Detected: " << stats_.corruption_detected << "\n";

    return oss.str();
  }

 private:
  MemoryForensicsTracker() = default;

  static uint64_t GetCurrentTimestampMs() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  }

  mutable mutex mu_;
  uint64_t next_alloc_id_{1};
  std::unordered_map<void*, std::vector<AllocationRecord>> allocation_history_
      TF_GUARDED_BY(mu_);
  std::unordered_map<void*, AllocationRecord> active_allocations_
      TF_GUARDED_BY(mu_);
  mutable MemoryStats stats_ TF_GUARDED_BY(mu_);
};

// MusaSubAllocator wraps musaMalloc/musaFree for use with TensorFlow's
// BFCAllocator. This replaces direct musaMalloc calls with a proper memory
// pooling strategy.
//
// Memory Coloring Implementation:
// - On allocation: Fill memory with magic pattern (0xDEADBEEF)
// - On free: Verify magic pattern integrity and fill with free pattern (0xFEEDFACE)
// - Track allocation history for forensics
// - Detect Use-After-Free and memory corruption
class MusaSubAllocator : public SubAllocator {
 public:
  MusaSubAllocator(int device_id, const std::vector<Visitor>& alloc_visitors,
                   const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors), device_id_(device_id) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    if (num_bytes == 0) {
      *bytes_received = 0;
      return nullptr;
    }

    // Ensure minimum alignment of 256 bytes (musaMalloc guarantee)
    // and respect the requested alignment from BFCAllocator
    size_t min_alignment = 256;
    if (alignment < min_alignment) {
      alignment = min_alignment;
    }

    // Round up allocation size to alignment boundary
    size_t alloc_size = (num_bytes + alignment - 1) & ~(alignment - 1);
    if (alloc_size < num_bytes) {
      // Overflow check
      return nullptr;
    }

    void* ptr = nullptr;
    musaSetDevice(device_id_);
    musaError_t err = musaMalloc(&ptr, alloc_size);
    if (err != musaSuccess) {
      LOG(WARNING) << "MusaSubAllocator: musaMalloc failed for " << alloc_size
                   << " bytes (alignment=" << alignment
                   << "): " << musaGetErrorString(err);
      return nullptr;
    }

    // Check alignment
    if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
      LOG(WARNING) << "MusaSubAllocator: musaMalloc returned unaligned pointer "
                   << ptr << " (requested alignment=" << alignment << ")";
      musaFree(ptr);
      return nullptr;
    }

    *bytes_received = alloc_size;

    // Apply memory coloring if enabled
    if (MemoryColoringConfig::Instance().enabled()) {
      ApplyMemoryColoring(ptr, alloc_size);
    }

    // Record allocation for forensics
    uint64_t tensor_id = MemoryForensicsTracker::Instance().RecordAllocation(
        ptr, alloc_size, device_id_);

    // Record telemetry event for allocation
    MUSA_TELEMETRY_ON_TENSOR_ALLOCATE(tensor_id, ptr, alloc_size, device_id_, 0);

    // Call visitor to track allocation
    VisitAlloc(ptr, device_id_, alloc_size);

    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      // Check for potential Use-After-Free before freeing
      if (MemoryColoringConfig::Instance().track_history()) {
        if (!MemoryForensicsTracker::Instance().IsAddressAllocated(ptr)) {
          LOG(WARNING) << "[MemoryForensics] Potential double-free or UAF detected"
                       << " for address " << ptr << " size=" << num_bytes;
          MemoryForensicsTracker::Instance().RecordCorruption("Double-free/UAF");
        }
      }

      // Verify memory coloring integrity before free
      if (MemoryColoringConfig::Instance().verify_on_free()) {
        VerifyMemoryColoring(ptr, num_bytes);
      }

      // Apply free pattern for UAF detection
      if (MemoryColoringConfig::Instance().enabled()) {
        ApplyFreePattern(ptr, num_bytes);
      }

      // Get tensor ID for telemetry before recording free
      auto history = MemoryForensicsTracker::Instance().GetAllocationHistory(ptr);
      uint64_t tensor_id = 0;
      if (!history.empty()) {
        tensor_id = history.back().alloc_id;
      }

      // Record free for forensics
      MemoryForensicsTracker::Instance().RecordFree(ptr, num_bytes, device_id_);

      // Record telemetry event for free
      MUSA_TELEMETRY_ON_TENSOR_FREE(tensor_id, ptr, num_bytes, device_id_);

      // Call visitor to track deallocation
      VisitFree(ptr, device_id_, num_bytes);

      musaSetDevice(device_id_);
      musaError_t err = musaFree(ptr);
      if (err != musaSuccess) {
        LOG(ERROR) << "MusaSubAllocator: musaFree failed: "
                   << musaGetErrorString(err);
      }
    }
  }

  bool SupportsCoalescing() const override { return true; }

 private:
  int device_id_;

  // Fill allocated memory with magic pattern
  void ApplyMemoryColoring(void* ptr, size_t size) {
    musaSetDevice(device_id_);

    // Fill entire allocation with allocation magic pattern
    // This helps detect uninitialized reads and provides baseline for corruption detection
    musaError_t err = musaMemset(ptr, kMusaAllocByte, size);
    if (err != musaSuccess) {
      LOG(WARNING) << "[MemoryColoring] Failed to apply allocation pattern: "
                   << musaGetErrorString(err);
    }

    VLOG(2) << "[MemoryColoring] Applied magic byte 0x" << std::hex
            << static_cast<int>(kMusaAllocByte)
            << std::dec << " to " << size << " bytes at " << ptr;
  }

  // Verify memory coloring integrity and apply free pattern
  void VerifyMemoryColoring(void* ptr, size_t size) {
    // Note: Reading device memory requires a device->host copy
    // For efficiency, we only verify a sample of the memory
    const size_t sample_size = std::min(size, static_cast<size_t>(4096));
    std::vector<uint32_t> host_buffer(sample_size / sizeof(uint32_t));

    musaSetDevice(device_id_);
    musaError_t err = musaMemcpy(host_buffer.data(), ptr, sample_size,
                                  musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
      LOG(WARNING) << "[MemoryColoring] Failed to read memory for verification: "
                   << musaGetErrorString(err);
      return;
    }

    // Check for signs of memory corruption
    // If we find the free magic in allocated memory, it indicates UAF
    for (size_t i = 0; i < host_buffer.size(); ++i) {
      if (host_buffer[i] == kMusaFreeMagic) {
        LOG(ERROR) << "[MemoryColoring] UAF detected! Found free magic 0x"
                   << std::hex << kMusaFreeMagic << std::dec
                   << " at offset " << i * sizeof(uint32_t)
                   << " in allocated memory at " << ptr;
        MemoryForensicsTracker::Instance().RecordCorruption("UAF detected");
        break;
      }
    }

    VLOG(2) << "[MemoryColoring] Verified memory at " << ptr;
  }

  // Apply free pattern to detect UAF
  void ApplyFreePattern(void* ptr, size_t size) {
    musaSetDevice(device_id_);

    // Fill with free magic pattern
    // If this pattern appears in a future allocation, we know UAF occurred
    musaError_t err = musaMemset(ptr, kMusaFreeByte, size);
    if (err != musaSuccess) {
      LOG(WARNING) << "[MemoryColoring] Failed to apply free pattern: "
                   << musaGetErrorString(err);
    }

    VLOG(2) << "[MemoryColoring] Applied free magic byte 0x" << std::hex
            << static_cast<int>(kMusaFreeByte)
            << std::dec << " to " << size << " bytes at " << ptr;
  }
};

// Helper functions for enabling/disabling memory coloring at runtime
inline void EnableMemoryColoring(bool enable = true) {
  MemoryColoringConfig::Instance().set_enabled(enable);
}

inline void EnableMemoryHistoryTracking(bool enable = true) {
  MemoryColoringConfig::Instance().set_track_history(enable);
}

inline void EnableMemoryVerification(bool enable = true) {
  MemoryColoringConfig::Instance().set_verify_on_free(enable);
}

inline bool IsMemoryColoringEnabled() {
  return MemoryColoringConfig::Instance().enabled();
}

// Get forensics report for an address
inline std::string GetMemoryForensicsReport(void* ptr) {
  return MemoryForensicsTracker::Instance().GenerateReport(ptr);
}

// Get current memory statistics
inline MemoryStats GetMemoryStats() {
  return MemoryForensicsTracker::Instance().GetStats();
}

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_ALLOCATOR_H_