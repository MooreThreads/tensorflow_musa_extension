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

#include "mu/device/host_caching_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mu/device/event_pool.h"

namespace tensorflow {
namespace musa {

namespace {

constexpr size_t kMinBucketBytes = 64 * 1024;  // 64 KiB minimum bucket.
constexpr size_t kDefaultMaxPoolBytes =
    size_t{2} * 1024 * 1024 * 1024;  // 2 GiB default cap.
// Max free-list depth per size class. Steady-state workloads typically
// cycle 1-2 buffers per class; a larger cap only hurts reclamation.
constexpr size_t kMaxFreePerClass = 8;

size_t NextPow2AtLeast(size_t n) {
  if (n <= kMinBucketBytes) return kMinBucketBytes;
  size_t s = kMinBucketBytes;
  while (s < n) s <<= 1;
  return s;
}

bool EnvTruthy(const char* name) {
  const char* e = std::getenv(name);
  if (e == nullptr || *e == '\0') return false;
  const std::string v(e);
  return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES" ||
         v == "on" || v == "ON";
}

size_t EnvMaxPoolBytes() {
  const char* e = std::getenv("TF_MUSA_HOST_ALLOC_MAX_POOL_MB");
  if (e == nullptr || *e == '\0') return kDefaultMaxPoolBytes;
  long long mb = std::atoll(e);
  if (mb <= 0) return kDefaultMaxPoolBytes;
  return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

}  // namespace

struct HostCachingAllocator::Impl {
  Impl() {
    disabled = EnvTruthy("TF_MUSA_DISABLE_HOST_CACHING");
    max_pool_bytes = EnvMaxPoolBytes();
    if (!disabled) {
      std::fprintf(
          stderr,
          "[MUSA] host caching allocator enabled (max_pool=%zu bytes)\n",
          max_pool_bytes);
      std::fflush(stderr);
    }
  }

  ~Impl() {
    // Best-effort teardown; by shutdown time MUSA runtime state may be
    // gone, in which case the driver calls silently error.
    std::lock_guard<std::mutex> lk(mu);
    for (auto& kv : free_lists) {
      for (void* p : kv.second) (void)musaFreeHost(p);
    }
    free_lists.clear();
    for (auto& kv : in_flight) {
      for (auto& entry : kv.second) {
        for (auto& ev : entry.events) {
          if (ev.second) EventPool::Instance().Release(ev.first, ev.second);
        }
        (void)musaFreeHost(entry.ptr);
      }
    }
    in_flight.clear();
  }

  // One in-flight buffer. `events` holds every outstanding event recorded
  // via RecordStream since the last Allocate; the buffer is eligible to
  // return to the free list only when ALL events complete. This handles
  // the rare-but-valid case where the same buffer is consumed by copies
  // on multiple streams.
  struct InFlight {
    void* ptr;
    size_t cls;
    std::vector<std::pair<int /*ordinal*/, musaEvent_t>> events;
  };

  // Meta for a pointer currently held by a caller. When the caller calls
  // Free(), `pending_events` is moved into a new InFlight entry if it is
  // non-empty, else the buffer returns to the free list immediately.
  struct Live {
    size_t cls;
    std::vector<std::pair<int /*ordinal*/, musaEvent_t>> pending_events;
  };

  mutable std::mutex mu;
  bool disabled = false;
  size_t max_pool_bytes = kDefaultMaxPoolBytes;

  // Counters. These cover only memory the allocator owns (in use with a
  // caller, on a free list, or parked on the in-flight list). Raw
  // passthrough allocations (disabled mode or cap-exceeded fallback
  // in the caller) are NOT included.
  size_t owned_bytes = 0;
  size_t in_use_bytes = 0;
  size_t peak_owned_bytes = 0;

  uint64_t alloc_requests = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t pool_cap_rejections = 0;
  uint64_t record_stream_count = 0;

  std::unordered_map<size_t /*cls*/, std::vector<void*>> free_lists;
  std::unordered_map<size_t /*cls*/, std::deque<InFlight>> in_flight;
  std::unordered_map<void* /*ptr*/, Live> live;

  void UpdatePeakLocked() {
    if (owned_bytes > peak_owned_bytes) peak_owned_bytes = owned_bytes;
  }

  // Returns completed in-flight entries to their free lists. Must be
  // called with `mu` held. For each InFlight entry we poll every
  // outstanding event; the buffer returns to the free list only once
  // every event reports complete.
  void DrainCompletedLocked() {
    for (auto& kv : in_flight) {
      const size_t cls = kv.first;
      auto& dq = kv.second;
      auto it = dq.begin();
      while (it != dq.end()) {
        // Remove any events that have completed; keep the rest in place.
        auto& evs = it->events;
        auto ev_it = evs.begin();
        while (ev_it != evs.end()) {
          musaError_t e = musaEventQuery(ev_it->second);
          if (e == musaSuccess) {
            EventPool::Instance().Release(ev_it->first, ev_it->second);
            ev_it = evs.erase(ev_it);
          } else {
            (void)musaGetLastError();
            ++ev_it;
          }
        }
        if (!evs.empty()) {
          ++it;
          continue;
        }
        // All events cleared: recycle.
        auto& fl = free_lists[cls];
        if (fl.size() < kMaxFreePerClass) {
          fl.push_back(it->ptr);
        } else {
          (void)musaFreeHost(it->ptr);
          owned_bytes -= cls;
        }
        it = dq.erase(it);
      }
    }
  }
};

// static
HostCachingAllocator& HostCachingAllocator::Instance() {
  // Intentionally never destroyed. When the plugin is split across multiple
  // shared libraries, static-local destructors at process exit fire after
  // some dependent libraries (libmusart, libtensorflow_framework) may have
  // been unloaded, which turned `musaFreeHost`/`musaEventDestroy` calls
  // during teardown into SIGSEGV. Leaking the instance costs nothing at
  // exit and avoids that class of bug.
  static HostCachingAllocator* kInst = new HostCachingAllocator();
  return *kInst;
}

HostCachingAllocator::HostCachingAllocator() : impl_(new Impl()) {}

HostCachingAllocator::~HostCachingAllocator() { delete impl_; }

bool HostCachingAllocator::Enabled() const { return !impl_->disabled; }

void* HostCachingAllocator::Allocate(size_t size) {
  if (size == 0) return nullptr;

  if (impl_->disabled) {
    void* p = nullptr;
    musaError_t e = musaHostAlloc(&p, size, musaHostAllocPortable);
    if (e != musaSuccess) {
      (void)musaGetLastError();
      return nullptr;
    }
    return p;
  }

  const size_t cls = NextPow2AtLeast(size);

  // Fast path: drain, then try the free list.
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    ++impl_->alloc_requests;
    impl_->DrainCompletedLocked();

    auto it = impl_->free_lists.find(cls);
    if (it != impl_->free_lists.end() && !it->second.empty()) {
      void* p = it->second.back();
      it->second.pop_back();
      impl_->live[p] = Impl::Live{cls, {}};
      impl_->in_use_bytes += cls;
      ++impl_->cache_hits;
      impl_->UpdatePeakLocked();
      return p;
    }

    if (impl_->owned_bytes + cls > impl_->max_pool_bytes) {
      ++impl_->pool_cap_rejections;
      return nullptr;
    }
    impl_->owned_bytes += cls;  // tentative reservation
  }

  // Slow path: grow the pool outside the lock.
  void* p = nullptr;
  musaError_t e = musaHostAlloc(&p, cls, musaHostAllocPortable);
  if (e != musaSuccess || p == nullptr) {
    (void)musaGetLastError();
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->owned_bytes -= cls;
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    ++impl_->cache_misses;
    impl_->live[p] = Impl::Live{cls, {}};
    impl_->in_use_bytes += cls;
    impl_->UpdatePeakLocked();
  }
  return p;
}

void HostCachingAllocator::Free(void* ptr) {
  if (ptr == nullptr) return;

  if (impl_->disabled) {
    (void)musaFreeHost(ptr);
    return;
  }

  std::lock_guard<std::mutex> lk(impl_->mu);
  auto it = impl_->live.find(ptr);
  if (it == impl_->live.end()) {
    // Not one of ours. Defensive fallback so foreign pinned buffers are
    // still released rather than leaked, and log once so misuse is
    // visible.
    static std::atomic<int> once{0};
    if (once.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::fprintf(stderr,
                   "[MUSA] HostCachingAllocator::Free(%p) ignored: pointer "
                   "not tracked (passthrough or foreign allocation).\n",
                   ptr);
      std::fflush(stderr);
    }
    (void)musaFreeHost(ptr);
    return;
  }
  const size_t cls = it->second.cls;
  auto pending = std::move(it->second.pending_events);
  impl_->live.erase(it);
  impl_->in_use_bytes -= cls;

  if (!pending.empty()) {
    // Buffer has outstanding DMA. Park it and its events.
    Impl::InFlight entry;
    entry.ptr = ptr;
    entry.cls = cls;
    entry.events = std::move(pending);
    impl_->in_flight[cls].push_back(std::move(entry));
    return;
  }

  auto& fl = impl_->free_lists[cls];
  if (fl.size() < kMaxFreePerClass) {
    fl.push_back(ptr);
  } else {
    (void)musaFreeHost(ptr);
    impl_->owned_bytes -= cls;
  }
}

void HostCachingAllocator::RecordStream(void* ptr, int ordinal,
                                        musaStream_t stream) {
  if (ptr == nullptr) return;
  if (impl_->disabled) return;

  musaEvent_t ev = nullptr;
  if (EventPool::Instance().Acquire(ordinal, &ev) != musaSuccess ||
      ev == nullptr) {
    (void)musaGetLastError();
    static std::atomic<int> once{0};
    if (once.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::fprintf(stderr,
                   "[MUSA] HostCachingAllocator::RecordStream: "
                   "EventPool::Acquire failed; buffer will be recycled "
                   "eagerly on Free.\n");
      std::fflush(stderr);
    }
    return;
  }
  if (musaEventRecord(ev, stream) != musaSuccess) {
    (void)musaGetLastError();
    EventPool::Instance().Release(ordinal, ev);
    return;
  }
  std::lock_guard<std::mutex> lk(impl_->mu);
  ++impl_->record_stream_count;
  auto it = impl_->live.find(ptr);
  if (it == impl_->live.end()) {
    // Pointer was freed concurrently (unlikely but allowed). Drop the
    // event; whatever synchronization the caller expected is their
    // problem now.
    EventPool::Instance().Release(ordinal, ev);
    return;
  }
  it->second.pending_events.emplace_back(ordinal, ev);
}

void HostCachingAllocator::ProcessEvents() {
  if (impl_->disabled) return;
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->DrainCompletedLocked();
}

void HostCachingAllocator::EmptyCache() {
  if (impl_->disabled) return;
  std::lock_guard<std::mutex> lk(impl_->mu);
  for (auto& kv : impl_->in_flight) {
    for (auto& entry : kv.second) {
      for (auto& ev : entry.events) {
        (void)musaEventSynchronize(ev.second);
        EventPool::Instance().Release(ev.first, ev.second);
      }
      (void)musaFreeHost(entry.ptr);
      impl_->owned_bytes -= entry.cls;
    }
    kv.second.clear();
  }
  for (auto& kv : impl_->free_lists) {
    for (void* p : kv.second) {
      (void)musaFreeHost(p);
      impl_->owned_bytes -= kv.first;
    }
    kv.second.clear();
  }
}

HostCachingAllocatorStats HostCachingAllocator::GetStats() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  HostCachingAllocatorStats s;
  s.in_use_bytes = impl_->in_use_bytes;
  s.cached_bytes = impl_->owned_bytes - impl_->in_use_bytes;
  s.total_reserved_bytes = impl_->owned_bytes;
  s.peak_bytes = impl_->peak_owned_bytes;
  s.alloc_requests = impl_->alloc_requests;
  s.cache_hits = impl_->cache_hits;
  s.cache_misses = impl_->cache_misses;
  s.pool_cap_rejections = impl_->pool_cap_rejections;
  s.record_stream_count = impl_->record_stream_count;
  return s;
}

}  // namespace musa
}  // namespace tensorflow
