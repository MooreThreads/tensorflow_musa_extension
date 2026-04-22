/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

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

// Implementation of the device caching allocator; see the header for the
// design overview. This file lives in libmusa_core.so so there is exactly
// one instance per ordinal across libmusa_plugin.so and the future
// tensorflow_musa._C pybind module.

#include "mu/device/caching_allocator.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorflow {
namespace musa {

namespace {

// Minimum allocation granularity. Rounding to 512 B matches the CUDA
// caching allocator's `kMinBlockSize` and plays well with L2 cache
// lines / typical tensor alignment constraints.
constexpr size_t kMinBlockSize = 512;

// Segments are obtained from `musaMalloc` in quanta of the larger of
// (requested rounded) and `kSegmentQuantum`. 2 MiB chunks are the
// historical "small" bucket in CUDA; keeping the same number makes
// profile comparisons meaningful across frameworks.
constexpr size_t kSegmentQuantum = 2 * 1024 * 1024;

// If splitting would leave a free remainder smaller than this, we keep
// the whole block in one piece. Avoids producing dust that no future
// allocation can fit.
constexpr size_t kMinSplitRemainder = 1024;

// Configurable-ish pool cap. The caching allocator only releases
// full segments from the driver inside `EmptyCache`, so runaway caches
// are a risk on long-running processes. This upper bound is a safety
// net, not a hard allocation limit: live allocations are always
// serviced even if they push the reserved bytes above the cap.
size_t PoolCapBytes() {
  const char* e = std::getenv("TF_MUSA_DEVICE_ALLOC_MAX_POOL_MB");
  if (e == nullptr || *e == '\0') {
    // 32 GiB default: well above any single-step working set but low
    // enough that a leak is noticeable.
    return size_t{32} * 1024 * 1024 * 1024;
  }
  long long v = std::atoll(e);
  if (v <= 0) return SIZE_MAX;
  return static_cast<size_t>(v) * 1024 * 1024;
}

size_t RoundUp(size_t n, size_t granularity) {
  return ((n + granularity - 1) / granularity) * granularity;
}

size_t SegmentSizeFor(size_t requested_rounded) {
  // Either a whole number of 2 MiB chunks, or the exact request size
  // for very large (multi-segment-worth) allocations. This avoids
  // wasting 100s of MiB when a single op asks for an odd-shaped tensor.
  if (requested_rounded <= kSegmentQuantum) return kSegmentQuantum;
  return RoundUp(requested_rounded, kSegmentQuantum);
}

bool EnvEquals(const char* name, const char* value) {
  const char* e = std::getenv(name);
  if (e == nullptr || *e == '\0') return false;
  return std::string(e) == value;
}

}  // namespace

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------
DeviceAllocatorBackend GetDeviceAllocatorBackend() {
  // Cache the decision in a function-local static so we re-read the env
  // exactly once per process. Allocator selection at runtime must be
  // consistent across allocate/deallocate.
  static const DeviceAllocatorBackend kBackend = []() {
    const char* e = std::getenv("TF_MUSA_DEVICE_ALLOCATOR");
    if (e == nullptr || *e == '\0') return DeviceAllocatorBackend::kCaching;
    const std::string v(e);
    if (v == "caching") return DeviceAllocatorBackend::kCaching;
    if (v == "passthrough") return DeviceAllocatorBackend::kPassthrough;
    std::fprintf(stderr,
                 "[MUSA] TF_MUSA_DEVICE_ALLOCATOR=%s not recognized, "
                 "falling back to caching.\n",
                 e);
    return DeviceAllocatorBackend::kCaching;
  }();
  return kBackend;
}

const char* DeviceAllocatorBackendName(DeviceAllocatorBackend b) {
  switch (b) {
    case DeviceAllocatorBackend::kCaching:
      return "caching";
    case DeviceAllocatorBackend::kPassthrough:
      return "passthrough";
  }
  return "unknown";
}

// -----------------------------------------------------------------------------
// Block and pool
// -----------------------------------------------------------------------------
struct Block {
  void* ptr = nullptr;
  size_t size = 0;
  int device = 0;
  bool allocated = false;

  // Segment-local doubly-linked list for address-ordered merging. Both
  // pointers are nullptr at the ends of a segment.
  Block* prev = nullptr;
  Block* next = nullptr;

  // True iff this block is the head of a segment (i.e. the pointer
  // returned by musaMalloc). Segments can only be released to the
  // driver via `EmptyCache` when `is_segment_head && !allocated && prev
  // == nullptr && next == nullptr`.
  bool is_segment_head = false;
};

struct BlockCmpBySize {
  bool operator()(Block* a, Block* b) const {
    if (a->size != b->size) return a->size < b->size;
    return reinterpret_cast<uintptr_t>(a->ptr) <
           reinterpret_cast<uintptr_t>(b->ptr);
  }
};

struct DeviceCachingAllocator::Impl {
  explicit Impl(int ord) : device(ord) {}

  // Guards every field below. Held across set_device for the short
  // critical section, released around `musaMalloc` / `musaFree`.
  mutable std::mutex mu;

  const int device;

  // Size-ordered free blocks for best-fit lookup.
  std::set<Block*, BlockCmpBySize> free_blocks;
  // Pointer → live Block. Populated on every Allocate, removed on Free.
  std::unordered_map<void*, Block*> active_blocks;

  // Stats counters.
  uint64_t in_use_bytes = 0;
  uint64_t reserved_bytes = 0;
  uint64_t peak_in_use_bytes = 0;
  uint64_t alloc_requests = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t oom_events = 0;
  uint64_t splits = 0;
  uint64_t merges = 0;
  uint64_t segments = 0;

  // Returns the smallest free block with size >= `min_size`, or nullptr.
  Block* FindFitLocked(size_t min_size) {
    // Use a transient key to binary-search into the size-ordered set.
    // ptr=nullptr ensures the key is the minimum among equal-size
    // entries, so lower_bound returns the smallest qualifying block.
    Block key;
    key.size = min_size;
    key.ptr = nullptr;
    auto it = free_blocks.lower_bound(&key);
    if (it == free_blocks.end()) return nullptr;
    return *it;
  }

  // Allocates a fresh segment from the driver. Lock may be released
  // across musaMalloc so we don't serialize other host threads.
  Block* AllocateSegmentUnlocked(size_t rounded_size) {
    const size_t seg_size = SegmentSizeFor(rounded_size);
    void* p = nullptr;
    musaError_t err = musaMalloc(&p, seg_size);
    if (err != musaSuccess || p == nullptr) {
      (void)musaGetLastError();
      return nullptr;
    }
    auto* b = new Block();
    b->ptr = p;
    b->size = seg_size;
    b->device = device;
    b->allocated = false;
    b->is_segment_head = true;
    return b;
  }

  // If `b` is bigger than requested by at least `kMinSplitRemainder`,
  // cut off the tail and return the head portion. The tail goes onto
  // the free list. `b` must not be in `free_blocks` when this is called.
  void MaybeSplitLocked(Block* b, size_t requested_rounded) {
    if (b->size < requested_rounded + kMinSplitRemainder) return;
    auto* tail = new Block();
    tail->ptr = static_cast<char*>(b->ptr) + requested_rounded;
    tail->size = b->size - requested_rounded;
    tail->device = b->device;
    tail->allocated = false;
    tail->is_segment_head = false;

    tail->prev = b;
    tail->next = b->next;
    if (b->next) b->next->prev = tail;
    b->next = tail;
    b->size = requested_rounded;

    free_blocks.insert(tail);
    ++splits;
  }

  // Merges `b` with its left and right neighbors if both exist and are
  // free. `b` must not be in `free_blocks` or `active_blocks` at entry.
  // After the call the resulting merged block may have absorbed one or
  // both neighbors; it is still not on the free list.
  Block* MergeNeighborsLocked(Block* b) {
    bool merged_any = false;
    if (b->prev != nullptr && !b->prev->allocated) {
      Block* p = b->prev;
      free_blocks.erase(p);
      p->size += b->size;
      p->next = b->next;
      if (b->next) b->next->prev = p;
      delete b;
      b = p;
      merged_any = true;
    }
    if (b->next != nullptr && !b->next->allocated) {
      Block* n = b->next;
      free_blocks.erase(n);
      b->size += n->size;
      b->next = n->next;
      if (n->next) n->next->prev = b;
      delete n;
      merged_any = true;
    }
    if (merged_any) ++merges;
    return b;
  }
};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

namespace {
struct PerDeviceRegistry {
  std::mutex mu;
  std::unordered_map<int, DeviceCachingAllocator*> per_device;
};

PerDeviceRegistry& Registry() {
  // Leaked on purpose, see HostCachingAllocator for the full rationale
  // (static destructors run after libmusart may be unloaded).
  static PerDeviceRegistry* r = new PerDeviceRegistry();
  return *r;
}
}  // namespace

DeviceCachingAllocator& DeviceCachingAllocator::For(int ordinal) {
  auto& r = Registry();
  std::lock_guard<std::mutex> lk(r.mu);
  auto it = r.per_device.find(ordinal);
  if (it != r.per_device.end()) return *it->second;
  auto* a = new DeviceCachingAllocator(ordinal);
  r.per_device.emplace(ordinal, a);
  return *a;
}

DeviceCachingAllocator::DeviceCachingAllocator(int ordinal)
    : impl_(new Impl(ordinal)) {}

DeviceCachingAllocator::~DeviceCachingAllocator() { delete impl_; }

void* DeviceCachingAllocator::Allocate(size_t size) {
  if (size == 0) return nullptr;
  const size_t rounded = RoundUp(size, kMinBlockSize);

  // Make sure our musaMalloc executes on the right device. The enclosing
  // TF callback has already done CachedSetDevice, but the allocator is
  // also used from the future pybind API where the context may differ.
  musaError_t e = musaSetDevice(impl_->device);
  if (e != musaSuccess) {
    (void)musaGetLastError();
    return nullptr;
  }

  Block* block = nullptr;
  bool is_cache_hit = false;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    ++impl_->alloc_requests;
    block = impl_->FindFitLocked(rounded);
    if (block != nullptr) {
      impl_->free_blocks.erase(block);
      impl_->MaybeSplitLocked(block, rounded);
      is_cache_hit = true;
    }
  }

  if (block == nullptr) {
    // Cache miss — call the driver outside the lock so other host
    // threads can keep making allocator progress.
    block = impl_->AllocateSegmentUnlocked(rounded);
    if (block == nullptr) {
      std::lock_guard<std::mutex> lk(impl_->mu);
      ++impl_->oom_events;
      return nullptr;
    }
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->reserved_bytes += block->size;
    ++impl_->segments;
    impl_->MaybeSplitLocked(block, rounded);
    ++impl_->cache_misses;
  }

  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    block->allocated = true;
    impl_->active_blocks.emplace(block->ptr, block);
    impl_->in_use_bytes += block->size;
    if (impl_->in_use_bytes > impl_->peak_in_use_bytes) {
      impl_->peak_in_use_bytes = impl_->in_use_bytes;
    }
    if (is_cache_hit) ++impl_->cache_hits;
  }

  // Respect the safety cap in a best-effort way: if we blew past it
  // with this allocation we trim free segments to bring things back.
  // This does not fail live requests.
  if (impl_->reserved_bytes > PoolCapBytes()) {
    EmptyCache();
  }

  return block->ptr;
}

void DeviceCachingAllocator::Free(void* ptr) {
  if (ptr == nullptr) return;
  std::lock_guard<std::mutex> lk(impl_->mu);
  auto it = impl_->active_blocks.find(ptr);
  if (it == impl_->active_blocks.end()) return;  // Not ours.
  Block* b = it->second;
  impl_->active_blocks.erase(it);
  impl_->in_use_bytes -= b->size;
  b->allocated = false;
  b = impl_->MergeNeighborsLocked(b);
  impl_->free_blocks.insert(b);
}

uint64_t DeviceCachingAllocator::EmptyCache() {
  std::vector<Block*> to_free;
  uint64_t released = 0;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (auto it = impl_->free_blocks.begin();
         it != impl_->free_blocks.end();) {
      Block* b = *it;
      // Only release full segments (head, no siblings, not allocated).
      if (b->is_segment_head && b->prev == nullptr && b->next == nullptr) {
        to_free.push_back(b);
        it = impl_->free_blocks.erase(it);
        released += b->size;
      } else {
        ++it;
      }
    }
    impl_->reserved_bytes -= released;
    if (!to_free.empty()) impl_->segments -= to_free.size();
  }

  // Free outside the lock.
  if (!to_free.empty()) {
    (void)musaSetDevice(impl_->device);
    for (Block* b : to_free) {
      musaError_t e = musaFree(b->ptr);
      (void)e;
      delete b;
    }
  }
  return released;
}

DeviceCachingAllocatorStats DeviceCachingAllocator::GetStats() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  DeviceCachingAllocatorStats s;
  s.in_use_bytes = impl_->in_use_bytes;
  s.reserved_bytes = impl_->reserved_bytes;
  s.cached_bytes = impl_->reserved_bytes >= impl_->in_use_bytes
                       ? impl_->reserved_bytes - impl_->in_use_bytes
                       : 0;
  s.peak_in_use_bytes = impl_->peak_in_use_bytes;
  s.alloc_requests = impl_->alloc_requests;
  s.cache_hits = impl_->cache_hits;
  s.cache_misses = impl_->cache_misses;
  s.oom_events = impl_->oom_events;
  s.splits = impl_->splits;
  s.merges = impl_->merges;
  s.segments = impl_->segments;
  return s;
}

void DeviceCachingAllocator::ResetPeakStats() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->peak_in_use_bytes = impl_->in_use_bytes;
}

}  // namespace musa
}  // namespace tensorflow
