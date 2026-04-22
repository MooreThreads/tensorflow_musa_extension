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

// Device-side caching allocator for MUSA memory.
//
// Rationale
// ---------
// Prior to this allocator the plugin relied on TF's PluggableDeviceBFC
// wrapper on top of raw `musaMalloc`/`musaFree`. That works correctly but
// has two problems for steady-state training:
//   * `musaMalloc` is a syscall-grade call (hundreds of microseconds)
//     and BFC only amortizes within the wrapper — a `tf.function` retrace
//     or a change in tensor shapes can punch through BFC's best-fit and
//     hit the driver again.
//   * BFC does not understand per-stream lifetime; its free-block
//     reclamation is synchronized via the single allocator lock, which
//     serializes ops that could otherwise reclaim across independent
//     streams.
//
// Design
// ------
// A per-device pool of ``Block``s keeps pre-allocated segments of MUSA
// memory organized as:
//   * a size-ordered ``std::set<Block*, BlockComparator>`` for best-fit
//     free-list lookup, and
//   * per-segment doubly-linked lists (``Block::prev`` / ``Block::next``)
//     for address-ordered merging when freed.
//
// ``Allocate(size)`` rounds the request up to a 512 B multiple, picks
// the smallest free block that fits, splits off the remainder if the
// leftover is large enough to be usable, and returns the head.
// ``Free(ptr)`` puts the block back on the free list, merging with its
// left/right neighbors when they are also free. OOM surfaces as
// ``nullptr``; callers (TF's SP_StreamExecutor.allocate via
// ``use_bfc_allocator=0``) treat that as a normal allocator failure.
//
// What is NOT in scope here (future work)
// ---------------------------------------
//   * Stream-ordered reuse (``stream_uses`` / ``record_stream`` +
//     ``musa_events``). TF's PluggableDevice contract guarantees that
//     deallocate is called only after the tensor is no longer in-flight,
//     so the MVP does not need per-stream event tracking. A future
//     commit adds it to support external ``RecordStream`` APIs.
//   * Expandable VMM segments (commit C5).
//   * Memory fraction, snapshot, OOM observer (commit C4).
//   * Small/large pool split. The MVP keeps a single pool per device;
//     if profiling shows churn near the 1 MiB boundary we can revisit.
//
// Safety & correctness
// --------------------
// * Allocations and frees are thread-safe under the allocator's mutex.
// * TF may call into the allocator concurrently from multiple host
//   threads (parallel copy engine, op pipeline). We take the lock for
//   every mutation of the per-device state but release it around
//   `musaMalloc` / `musaFree` so the driver does not serialize on us.
// * The allocator NEVER frees live blocks. Segments only go back to the
//   driver on ``EmptyCache()`` (unsplit, unused segments only). TF
//   calling ``deallocate`` before its own sync is a programming error
//   that we cannot detect here — same contract as CUDA PluggableDevice.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_CACHING_ALLOCATOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_CACHING_ALLOCATOR_H_

#include <musa_runtime.h>

#include <cstddef>
#include <cstdint>

namespace tensorflow {
namespace musa {

// Lightweight snapshot of allocator counters. All numbers are in bytes
// unless otherwise noted. Kept minimal for the C3 MVP; commit C4 will
// extend with per-class histograms and fraction-limit signals.
struct DeviceCachingAllocatorStats {
  // Total bytes currently handed out to callers.
  uint64_t in_use_bytes = 0;
  // Total bytes currently held by the allocator (reserved - freed segs).
  uint64_t reserved_bytes = 0;
  // Cached bytes = reserved - in-use, i.e. what could serve a future
  // Allocate without hitting musaMalloc.
  uint64_t cached_bytes = 0;
  // High-water mark of in_use_bytes over the lifetime (or since
  // ResetPeakStats).
  uint64_t peak_in_use_bytes = 0;

  uint64_t alloc_requests = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;  // Required musaMalloc.
  uint64_t oom_events = 0;
  uint64_t splits = 0;    // Block split into a used head + free tail.
  uint64_t merges = 0;    // Block merged with at least one neighbor.
  uint64_t segments = 0;  // Number of live segments obtained from driver.
};

// Runtime policy toggles for the device allocator. Set by the env var
// `TF_MUSA_DEVICE_ALLOCATOR`:
//   * "caching"      (default) — use the caching allocator below.
//   * "passthrough"  — raw musaMalloc/musaFree, no caching, no BFC on
//                      top. Useful for correctness bisection.
// Pair this with the BFC toggle in device_register.cc: when
// `Backend() == kPassthrough` we still want TF's PluggableDeviceBFC to
// pool raw musaMalloc allocations, matching the pre-C3 behavior.
enum class DeviceAllocatorBackend {
  kCaching,
  kPassthrough,
};

// Reads the env var once and caches the answer. Cheap to call from
// callbacks on the hot path.
DeviceAllocatorBackend GetDeviceAllocatorBackend();

// Returns a human-readable name for the active backend.
const char* DeviceAllocatorBackendName(DeviceAllocatorBackend b);

class DeviceCachingAllocator {
 public:
  // One allocator instance per device ordinal, lazily created. Thread-safe.
  static DeviceCachingAllocator& For(int ordinal);

  // Returns a `size`-byte device pointer on the owning ordinal. The
  // underlying segment lifetime is owned by the allocator; the caller
  // should release it with `Free(ptr)` (not `musaFree`).
  //
  // Returns nullptr on OOM. On nullptr the caller (TF's allocate
  // callback) should treat the request as failed; TF will propagate
  // through ResourceExhausted up the op scheduler.
  void* Allocate(size_t size);

  // Releases `ptr` previously returned by Allocate. Merges with address
  // neighbors when possible. Passing a pointer this allocator did not
  // hand out is a no-op.
  void Free(void* ptr);

  // Releases every *fully-free* segment back to the driver. Segments
  // that still have any live block stay; this is used for low-memory
  // fallback and for Python `empty_cache()`.
  //
  // Returns the number of bytes released.
  uint64_t EmptyCache();

  // Snapshot the current counters.
  DeviceCachingAllocatorStats GetStats() const;

  // Reset `peak_in_use_bytes` to the current in_use_bytes value.
  void ResetPeakStats();

 private:
  DeviceCachingAllocator(int ordinal);
  ~DeviceCachingAllocator();
  DeviceCachingAllocator(const DeviceCachingAllocator&) = delete;
  DeviceCachingAllocator& operator=(const DeviceCachingAllocator&) = delete;

  struct Impl;
  Impl* const impl_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_CACHING_ALLOCATOR_H_
