#include "musa_allocator.h"

#include <algorithm>
#include <cstring>

namespace tensorflow {
namespace musa {

// Out-of-class definition for static constexpr members
constexpr size_t MusaBFCAllocator::kMinAllocationSize;
constexpr size_t MusaBFCAllocator::kMaxPoolSize;
constexpr size_t MusaBFCAllocator::kAllocationAlignment;

// Simple Pool-based Allocator Implementation
// Uses power-of-2 size classes for efficient reuse

MusaBFCAllocator::MusaBFCAllocator(int device_id, size_t total_memory)
    : device_id_(device_id), pool_bytes_(0), allocated_bytes_(0) {
  VLOG(1) << "MUSA BFC Allocator created for device " << device_id;
}

MusaBFCAllocator::~MusaBFCAllocator() {
  // Free all pooled memory
  // Return all allocated pointers to pool for cleanup
  for (auto& pair : allocated_sizes_) {
    void* ptr = pair.first;
    size_t size_class = pair.second;
    pools_[size_class].push(ptr);
  }
  allocated_sizes_.clear();

  // Free all pooled memory
  for (auto& pair : pools_) {
    auto& queue = pair.second;
    while (!queue.empty()) {
      void* ptr = queue.front();
      queue.pop();
      musaSetDevice(device_id_);
      musaFree(ptr);
    }
  }
  pools_.clear();

  // Free all raw allocations
  for (void* ptr : musa_allocations_) {
    musaSetDevice(device_id_);
    musaFree(ptr);
  }

  VLOG(1) << "MUSA BFC Allocator destroyed. Allocs: " << num_allocs_
          << ", Deallocs: " << num_deallocs_
          << ", Pool hits: " << num_pool_hits_
          << ", Misses: " << num_pool_misses_;
}

size_t MusaBFCAllocator::RoundedBytes(size_t bytes) const {
  if (bytes == 0) return kMinAllocationSize;

  // Round up to minimum allocation size
  size_t rounded = std::max(bytes, kMinAllocationSize);

  // Round up to alignment
  size_t mask = kAllocationAlignment - 1;
  rounded = (rounded + mask) & ~mask;

  return rounded;
}

size_t MusaBFCAllocator::GetSizeClass(size_t bytes) const {
  // Use power-of-2 size classes
  size_t size = RoundedBytes(bytes);

  // Round up to next power of 2 for size class
  size_t power = 1;
  while (power < size) {
    power <<= 1;
  }

  // Cap at max pool size
  return std::min(power, kMaxPoolSize);
}

void* MusaBFCAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  if (num_bytes == 0) return nullptr;

  size_t size_class = GetSizeClass(num_bytes);

  // For very large allocations, use direct allocation
  if (size_class > kMaxPoolSize / 2) {
    musaSetDevice(device_id_);
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, num_bytes);
    if (err != musaSuccess) {
      LOG(ERROR) << "MUSA BFC Allocator: Direct allocation failed for "
                 << num_bytes << " bytes: " << musaGetErrorString(err);
      return nullptr;
    }

    mutex_lock l(mu_);
    musa_allocations_.push_back(ptr);
    allocated_bytes_ += num_bytes;
    num_allocs_++;

    VLOG(2) << "BFCAllocator: Direct alloc " << num_bytes << " bytes at "
            << ptr;
    return ptr;
  }

  // Try to get from pool
  void* ptr = nullptr;
  {
    mutex_lock l(mu_);

    auto it = pools_.find(size_class);
    if (it != pools_.end() && !it->second.empty()) {
      // Reuse from pool
      ptr = it->second.front();
      it->second.pop();
      num_pool_hits_++;
    }
  }

  if (ptr) {
    // Found in pool
    mutex_lock l(mu_);
    allocated_sizes_[ptr] = size_class;
    allocated_bytes_ += size_class;
    num_allocs_++;

    VLOG(2) << "BFCAllocator: Reused " << size_class << " bytes from pool at "
            << ptr;
    return ptr;
  }

  // Not in pool, allocate new memory
  musaSetDevice(device_id_);
  ptr = nullptr;
  musaError_t err = musaMalloc(&ptr, size_class);
  if (err != musaSuccess) {
    LOG(ERROR) << "MUSA BFC Allocator: musaMalloc failed for " << size_class
               << " bytes: " << musaGetErrorString(err);
    return nullptr;
  }

  {
    mutex_lock l(mu_);
    allocated_sizes_[ptr] = size_class;
    allocated_bytes_ += size_class;
    pool_bytes_ += size_class;
    num_allocs_++;
    num_pool_misses_++;
  }

  VLOG(2) << "BFCAllocator: Allocated " << size_class << " bytes at " << ptr;
  return ptr;
}

void MusaBFCAllocator::DeallocateRaw(void* ptr) {
  if (!ptr) return;

  size_t size_class = 0;
  bool is_tracked = false;
  bool is_raw = false;

  {
    mutex_lock l(mu_);

    // Check if this is a tracked allocation
    auto it = allocated_sizes_.find(ptr);
    if (it != allocated_sizes_.end()) {
      size_class = it->second;
      allocated_sizes_.erase(it);
      allocated_bytes_ -= size_class;
      num_deallocs_++;
      is_tracked = true;
    } else {
      // Not tracked - might be a direct allocation
      auto raw_it =
          std::find(musa_allocations_.begin(), musa_allocations_.end(), ptr);
      if (raw_it != musa_allocations_.end()) {
        musa_allocations_.erase(raw_it);
        allocated_bytes_ = (allocated_bytes_ > 0) ? allocated_bytes_ - 1 : 0;
        num_deallocs_++;
        is_raw = true;
      } else {
        LOG(WARNING)
            << "MUSA BFC Allocator: Attempting to free untracked pointer "
            << ptr;
        return;
      }
    }
  }  // Mutex released here

  if (is_tracked) {
    // Return to pool for reuse
    {
      mutex_lock l(mu_);
      pools_[size_class].push(ptr);
    }
    VLOG(2) << "BFCAllocator: Returned " << size_class << " bytes to pool at "
            << ptr;
  } else if (is_raw) {
    // Free directly
    musaSetDevice(device_id_);
    musaFree(ptr);
    VLOG(2) << "BFCAllocator: Freed direct allocation at " << ptr;
  }
}

}  // namespace musa
}  // namespace tensorflow
