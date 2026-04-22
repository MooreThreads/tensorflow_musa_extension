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

// Caching allocator for pinned (page-locked) host memory.
//
// Why
// ---
// `musaHostAlloc` / `musaFreeHost` are syscall-grade expensive: each pins
// pages in the kernel's DMA map. TensorFlow's PluggableDevice calls
// `host_memory_allocate` frequently (every H2D/D2H staging buffer, every
// `tf.constant` feed copied through HostMemory tensors, etc.), and its
// own BFC layer does NOT cache host buffers. Without this allocator we
// paid the full pin cost on every call.
//
// Design
// ------
// * Size-class bucketed free list: requests get rounded up to the next
//   power of two with a 64 KiB floor. For each bucket we keep a LIFO of
//   ready-to-reuse buffers. Because most workloads cycle through a small
//   set of tensor shapes, steady-state hit rate is effectively 100%.
// * Stream-ordered reuse via `EventPool`: callers who finish writing into
//   a pinned buffer but still have an outstanding MUSA DMA on it call
//   `RecordStream(ptr, ordinal, stream)`; we stash an event and only
//   return the buffer to the free list once the event retires. Buffers
//   not marked with `RecordStream` go back to the free list immediately
//   on `Free` (that is the expected case for `host_memory_deallocate`,
//   where TF owns the synchronization).
// * Pinned bytes are capped globally via `TF_MUSA_HOST_ALLOC_MAX_POOL_MB`
//   (default 2048). Once the cap is hit `Allocate` returns nullptr, and
//   the caller's `host_memory_allocate` wrapper falls back to a one-shot
//   `musaHostAlloc` so we still serve the request (just without caching).
// * Disabled by setting `TF_MUSA_DISABLE_HOST_CACHING=1`; that forces the
//   old raw `musaHostAlloc` / `musaFreeHost` behavior.
//
// Relationship to other pools
// ---------------------------
// * `PinnedStagingPool` (in `musa_se_callbacks.cc`) previously kept its
//   own size-classed pool of pinned staging buffers for H2D copies from
//   pageable sources. It now delegates to this allocator so we have a
//   single source of truth for cached pinned memory. Its in-flight
//   tracking is replaced by `RecordStream`.
// * `HostPinCache` is orthogonal: it calls `musaHostRegister` on
//   caller-owned buffers. It does not allocate and so does not interact
//   with this allocator beyond the fact that buffers we hand out will
//   report as pinned via `musaHostGetFlags`.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_HOST_CACHING_ALLOCATOR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_HOST_CACHING_ALLOCATOR_H_

#include <musa_runtime.h>

#include <cstddef>
#include <cstdint>

namespace tensorflow {
namespace musa {

struct HostCachingAllocatorStats {
  // Bytes currently handed out to callers and not yet freed.
  uint64_t in_use_bytes = 0;
  // Bytes currently held as cached (free list + in-flight).
  uint64_t cached_bytes = 0;
  // Total pinned bytes obtained from `musaHostAlloc` over the lifetime.
  uint64_t total_reserved_bytes = 0;
  // High-water mark of `in_use_bytes + cached_bytes`.
  uint64_t peak_bytes = 0;

  uint64_t alloc_requests = 0;       // Total Allocate() calls.
  uint64_t cache_hits = 0;           // Served from free list.
  uint64_t cache_misses = 0;         // Required `musaHostAlloc`.
  uint64_t pool_cap_rejections = 0;  // Hit `max_pool_bytes_` cap.
  uint64_t record_stream_count = 0;  // `RecordStream` invocations.
};

class HostCachingAllocator {
 public:
  static HostCachingAllocator& Instance();

  // Disabled when `TF_MUSA_DISABLE_HOST_CACHING=1`. When disabled,
  // `Allocate` and `Free` just proxy to `musaHostAlloc` / `musaFreeHost`,
  // `RecordStream` is a no-op, and the caller pays the full pin cost per
  // allocation. Useful for A/B correctness debugging.
  bool Enabled() const;

  // Returns a pinned host buffer at least `size` bytes large. Returns
  // nullptr on failure (including hitting the pool cap while no cached
  // buffer is available). The returned pointer is valid until `Free` is
  // called.
  //
  // Thread-safe.
  void* Allocate(size_t size);

  // Releases a pointer previously returned by `Allocate`. If `RecordStream`
  // was called on this pointer since the last `Allocate`, the buffer is
  // parked on the in-flight list and will be drained back to the free
  // list on the next `Allocate` (or manual `ProcessEvents`) after every
  // recorded event reports complete. Otherwise the buffer goes straight
  // to the free list.
  //
  // `ptr` must have come from this allocator's `Allocate`. Passing any
  // other pointer has undefined behavior (we will try to look it up, find
  // nothing, and either drop the pointer or fall through to musaFreeHost
  // when disabled).
  //
  // Thread-safe.
  void Free(void* ptr);

  // Marks `ptr` as busy on `stream` of `ordinal`: the next `Free(ptr)`
  // call will defer reuse until an event recorded now on that stream
  // completes. Safe to call multiple times; all recorded events must
  // complete before the buffer is recycled.
  //
  // Calls allocate an event from `EventPool` (which is cheap). On event
  // creation failure the function silently returns; the buffer is then
  // treated as ready-to-reuse on `Free`, which is safe ONLY if the
  // caller still synchronizes separately. Our in-tree callers always
  // hold another synchronization path (or are OK to `Free` immediately),
  // so this is acceptable.
  void RecordStream(void* ptr, int ordinal, musaStream_t stream);

  // Polls all in-flight events once and returns completed buffers to the
  // free list. Callers don't normally need to invoke this (Allocate
  // drains before serving), but it's useful for tests and for external
  // tools that want to release memory pressure.
  void ProcessEvents();

  // Releases every cached buffer (both free list and any in-flight
  // entries whose events have completed). Blocks on any outstanding
  // events first.
  void EmptyCache();

  // Snapshots the current stats counters. Thread-safe.
  HostCachingAllocatorStats GetStats() const;

 private:
  HostCachingAllocator();
  ~HostCachingAllocator();
  HostCachingAllocator(const HostCachingAllocator&) = delete;
  HostCachingAllocator& operator=(const HostCachingAllocator&) = delete;

  struct Impl;
  Impl* const impl_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_HOST_CACHING_ALLOCATOR_H_
