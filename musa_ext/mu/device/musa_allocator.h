#ifndef TENSORFLOW_MUSA_ALLOCATOR_H_
#define TENSORFLOW_MUSA_ALLOCATOR_H_

#include <musa_runtime.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

// Simple Pool-based Allocator for MUSA
// Uses a straightforward pool-per-size-class approach
class MusaBFCAllocator : public Allocator {
 public:
  explicit MusaBFCAllocator(int device_id, size_t total_memory = 0);
  ~MusaBFCAllocator() override;

  std::string Name() override { return "musa_bfc_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  // Memory statistics
  size_t AllocatedSize() const { return allocated_bytes_; }
  size_t AvailableSize() const { return pool_bytes_ - allocated_bytes_; }

 private:
  // Size class configuration
  static constexpr size_t kMinAllocationSize = 256;
  static constexpr size_t kMaxPoolSize = 1ULL << 30;  // 1GB max per pool
  static constexpr size_t kAllocationAlignment = 256;

  // Round up to alignment
  size_t RoundedBytes(size_t bytes) const;

  // Get size class for allocation
  size_t GetSizeClass(size_t bytes) const;

  int device_id_;
  size_t pool_bytes_;
  size_t allocated_bytes_;

  // Protects all mutable state
  mutable mutex mu_;

  // Pool for each size class
  // Key: size class (rounded allocation size)
  // Value: queue of available pointers
  std::unordered_map<size_t, std::queue<void*>> pools_;

  // Track which size class each allocated pointer belongs to
  std::unordered_map<void*, size_t> allocated_sizes_;

  // Track original MUSA allocations for cleanup
  std::vector<void*> musa_allocations_;

  // Statistics
  size_t num_allocs_ = 0;
  size_t num_deallocs_ = 0;
  size_t num_pool_hits_ = 0;
  size_t num_pool_misses_ = 0;
};

// Legacy raw allocator for comparison/testing
class MusaRawAllocator : public Allocator {
 public:
  explicit MusaRawAllocator(int device_id) : device_id_(device_id) {}

  ~MusaRawAllocator() override = default;

  std::string Name() override { return "musa_raw_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes == 0) return nullptr;

    musaSetDevice(device_id_);

    size_t target_alignment = std::max((size_t)256, alignment);

    // Check for overflow before calculation
    if (num_bytes > std::numeric_limits<size_t>::max() - target_alignment) {
      LOG(ERROR) << "MUSA allocator: allocation size overflow: " << num_bytes;
      return nullptr;
    }

    size_t alloc_bytes = (num_bytes + target_alignment - 1) / target_alignment *
                         target_alignment;

    // Check for overflow after adding padding
    if (alloc_bytes > std::numeric_limits<size_t>::max() - 256) {
      LOG(ERROR) << "MUSA allocator: allocation size overflow after padding: "
                 << alloc_bytes;
      return nullptr;
    }
    alloc_bytes += 256;

    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, alloc_bytes);
    if (err != musaSuccess) {
      LOG(ERROR) << "MUSA allocator: musaMalloc failed: "
                 << musaGetErrorString(err) << " size: " << alloc_bytes;
      return nullptr;
    }
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr) {
      musaSetDevice(device_id_);
      musaError_t err = musaFree(ptr);
      if (err != musaSuccess) {
        LOG(ERROR) << "MUSA allocator: musaFree failed: "
                   << musaGetErrorString(err);
      }
    }
  }

 private:
  int device_id_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_ALLOCATOR_H_
