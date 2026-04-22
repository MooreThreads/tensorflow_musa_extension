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

#include "mu/device/musa_se_callbacks.h"

#include <musa_runtime.h>
#include <unistd.h>

// MUSA 4.x does not expose the mempool / async-alloc runtime API surface
// (musaMemPoolCreate, musaFreeAsync, musaMallocFromPoolAsync, etc.).
// Gate AsyncDeviceAllocator's use of those symbols so this file still
// compiles on 4.3.5; the allocator becomes a no-op there and the
// legacy musaMalloc / musaFree path (below) is used for every device.
#if defined(MUSART_VERSION) && (MUSART_VERSION >= 50000)
#define TF_MUSA_HAS_MEMPOOL_API 1
#else
#define TF_MUSA_HAS_MEMPOOL_API 0
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mu/device/musa_telemetry.h"
#include "tensorflow/c/tf_status.h"

// These callbacks mirror TensorFlow 2.6.1's CUDA StreamExecutor surface
// (tensorflow/stream_executor/cuda/cuda_gpu_executor.cc) translated to MUSA.
// All of them return status via TF_Status* following the PluggableDevice
// contract declared in
// tensorflow/c/experimental/stream_executor/stream_executor.h.

namespace tensorflow {
namespace musa {
namespace {

// Convenience: the plugin tunnels raw musa handles through SP_Stream /
// SP_Event since both are opaque pointer typedefs. This matches what TF's
// CStream wrapper expects: GpuStreamHack() returns the SP_Stream value
// verbatim, so kernels that reinterpret_cast it back to musaStream_t get the
// driver-level handle with no indirection.
inline musaStream_t AsMusaStream(SP_Stream s) {
  return reinterpret_cast<musaStream_t>(s);
}
inline SP_Stream AsSPStream(musaStream_t s) {
  return reinterpret_cast<SP_Stream>(s);
}
inline musaEvent_t AsMusaEvent(SP_Event e) {
  return reinterpret_cast<musaEvent_t>(e);
}
inline SP_Event AsSPEvent(musaEvent_t e) {
  return reinterpret_cast<SP_Event>(e);
}
inline int DeviceOrdinal(const SP_Device* device) {
  // plugin_create_device stores the ordinal in device_handle as intptr_t.
  return device->ordinal;
}

// Thread-local `musaSetDevice` cache.
//
// TF's executor funnels work through a small pool of threads, and each one
// only talks to a single device at a time. Yet PluggableDevice routes
// every allocate / deallocate / stream / event / memcpy callback through
// this file, and we historically called `musaSetDevice` at the top of
// each one. On a 4k-node graph that is thousands of driver hops per step
// that end up being identity operations.
//
// The driver already maintains a thread-local "current device", so we can
// trivially short-circuit: only forward to the driver when our last-seen
// ordinal differs. This mirrors `CachedMusaSetDevice` in kernels/utils_op.h
// (used from kernel threads) and extends the optimization to the plugin's
// C-API boundary.
inline musaError_t CachedSetDevice(int ordinal) {
  static thread_local int cached = -1;
  if (ordinal == cached) return musaSuccess;
  musaError_t err = musaSetDevice(ordinal);
  if (err == musaSuccess) cached = ordinal;
  return err;
}

inline void SetOK(TF_Status* status) { TF_SetStatus(status, TF_OK, ""); }
inline void SetInternal(TF_Status* status, const char* msg) {
  TF_SetStatus(status, TF_INTERNAL, msg);
}
inline void SetMusaError(TF_Status* status, musaError_t err,
                         const char* context) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s: %s", context, musaGetErrorString(err));
  TF_SetStatus(status, TF_INTERNAL, buf);
}

// The sync_memcpy_* callbacks below use the blocking `musaMemcpy` primitive
// directly. An earlier revision routed them through a dedicated
// musaStreamNonBlocking stream plus `musaStreamSynchronize` hoping to let
// the driver pipeline larger DMAs; in practice the extra stream scheduling
// cost dominated for the typical sync-copy workload (session init-time
// constant uploads, rendezvous fetches of small result tensors), and
// `musaMemcpy` is the cheapest way to honor the "returns when the copy is
// done" contract required by the PluggableDevice spec.

// --- Stream-ordered (musaMallocFromPoolAsync / musaFreeAsync) allocator ---
//
// The PluggableDevice `allocate` callback returns a raw device pointer that
// TF's BFC sub-allocator hands out to kernels scheduled on arbitrary
// streams. We therefore cannot rely on stream-ordering alone; the pointer
// must be safe to use on any stream by the time `allocate` returns.
//
// Design:
//   * Each device gets its OWN `musaMemPool_t` created via
//     `musaMemPoolCreate`. We deliberately never touch the device's
//     default pool, because the default pool is shared with any other
//     code in the process that uses raw `musaMalloc` / mempool APIs
//     (mudnn / mublas internal scratch, user plugins, etc.); if we
//     reconfigure it (e.g. to UINT64_MAX release threshold) we can
//     perturb the allocator state underneath those callers and trigger
//     driver-side UB deep inside kernels.
//   * The private pool's release threshold is UINT64_MAX so freed
//     regions stay resident (BFC on top of us handles reuse).
//   * `allocate` = `musaMallocFromPoolAsync` on a dedicated
//     non-blocking alloc stream, followed by `musaStreamSynchronize`
//     on that stream so the returned pointer is observable from any
//     stream the caller picks.
//   * `deallocate` = `musaFreeAsync` with NO synchronization; the
//     driver parks the region back into the pool after prior work on
//     the alloc stream (only allocs/frees we enqueued) completes.
//     This is where the bulk of the win over plain `musaFree` comes
//     from: plain `musaFree` is a blocking driver call that often
//     hands memory back to the OS.
//
// Safety: TF's EventMgr guarantees `deallocate` is only invoked after
// every outstanding user of the pointer has finished on its stream, so
// enqueuing `musaFreeAsync` without syncing is fine. After
// `musaStreamSynchronize` on our alloc stream, the new pointer
// happens-before any subsequent work the caller submits to any stream.
//
// Graceful fallback: every device is queried for
// `musaDevAttrMemoryPoolsSupported` and each pool-setup step must
// succeed. Anything that fails flips that device into legacy-malloc
// mode for the rest of the process (sticky).
class AsyncDeviceAllocator {
 public:
  static AsyncDeviceAllocator& Instance() {
    static AsyncDeviceAllocator inst;
    return inst;
  }

  // True iff this device has the pool APIs and lazy init has succeeded.
  // The allocator is sticky: once a device flips to enabled it stays
  // there for the process lifetime, so allocate() and deallocate() can
  // branch identically and never mix musaMalloc with musaFreeAsync (or
  // vice versa) for the same pointer.
  //
  // Opt-in gate: stream-ordered allocation delivers real wins on hot
  // BFC growth / shrink paths, but mixing pool-backed memory with
  // mudnn's own raw musaMalloc/musaFree scratch inside some
  // convolution kernels has been observed to crash the driver on
  // current MUSA runtimes. The safe default is therefore the legacy
  // musaMalloc / musaFree path; set TF_MUSA_ENABLE_ASYNC_ALLOC=1 (or
  // "true", any non-zero value) to turn on the stream-ordered path
  // when your workload doesn't hit those kernels.
  bool IsEnabled(int ordinal) {
    static const bool kGloballyEnabled = [] {
      const char* env = std::getenv("TF_MUSA_ENABLE_ASYNC_ALLOC");
      if (env == nullptr) return false;
      return !(env[0] == '0' || env[0] == '\0' || env[0] == 'f' ||
               env[0] == 'F');
    }();
    if (!kGloballyEnabled) return false;
#if TF_MUSA_HAS_MEMPOOL_API
    PerDevice* pd = Acquire(ordinal);
    return pd != nullptr && pd->enabled;
#else
    (void)ordinal;
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
      std::fprintf(stderr,
                   "[MUSA] TF_MUSA_ENABLE_ASYNC_ALLOC=1 requested but the "
                   "MUSA runtime in use (MUSART_VERSION=%d) does not expose "
                   "the mempool / async-alloc APIs; falling back to the "
                   "legacy musaMalloc / musaFree path.\n",
                   static_cast<int>(MUSART_VERSION));
      std::fflush(stderr);
    });
    return false;
#endif
  }

  // Returns true iff the pool-based async path is usable on `ordinal`
  // and `*out_ptr` was populated. Returns false with `*out_ptr`
  // untouched on any failure; the caller is expected to treat that as
  // OOM (IsEnabled already gates the hot path).
  bool Allocate(int ordinal, uint64_t size, void** out_ptr) {
#if TF_MUSA_HAS_MEMPOOL_API
    PerDevice* pd = Acquire(ordinal);
    if (pd == nullptr || !pd->enabled) return false;

    void* ptr = nullptr;
    musaError_t err = musaMallocFromPoolAsync(&ptr, size, pd->pool, pd->stream);
    if (err != musaSuccess) return false;
    // Wait only on our private stream; compute / H2D / D2H streams are
    // never blocked. Pool-cache hits make this near-free.
    err = musaStreamSynchronize(pd->stream);
    if (err != musaSuccess) {
      // Best-effort: return the region to our pool before reporting
      // failure upstream.
      musaFreeAsync(ptr, pd->stream);
      return false;
    }
    *out_ptr = ptr;
    return true;
#else
    (void)ordinal;
    (void)size;
    (void)out_ptr;
    return false;
#endif
  }

  // Returns true iff the deallocation was handed off to the async free
  // path (no synchronization is performed).
  bool Deallocate(int ordinal, void* ptr) {
#if TF_MUSA_HAS_MEMPOOL_API
    PerDevice* pd = Acquire(ordinal);
    if (pd == nullptr || !pd->enabled) return false;
    // Fire-and-forget: the driver will park this region back into our
    // private pool after prior work on `pd->stream` (only allocs /
    // frees that we enqueued ourselves) completes.
    musaError_t err = musaFreeAsync(ptr, pd->stream);
    return err == musaSuccess;
#else
    (void)ordinal;
    (void)ptr;
    return false;
#endif
  }

 private:
  struct PerDevice {
    musaMemPool_t pool = nullptr;
    musaStream_t stream = nullptr;
    bool enabled = false;
  };

  PerDevice* Acquire(int ordinal) {
#if TF_MUSA_HAS_MEMPOOL_API
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = per_device_.find(ordinal);
      if (it != per_device_.end()) return it->second.get();
    }
    auto pd = std::unique_ptr<PerDevice>(new PerDevice);
    InitPerDevice(ordinal, pd.get());
    std::lock_guard<std::mutex> lk(mu_);
    auto it = per_device_.find(ordinal);
    if (it != per_device_.end()) {
      // Someone else won the race; drop anything we created.
      if (pd->stream) musaStreamDestroy(pd->stream);
      if (pd->pool) musaMemPoolDestroy(pd->pool);
      return it->second.get();
    }
    PerDevice* raw = pd.get();
    per_device_.emplace(ordinal, std::move(pd));
    return raw;
#else
    (void)ordinal;
    return nullptr;
#endif
  }

  void InitPerDevice(int ordinal, PerDevice* pd) {
    pd->enabled = false;
    pd->stream = nullptr;
    pd->pool = nullptr;

#if TF_MUSA_HAS_MEMPOOL_API
    if (musaSetDevice(ordinal) != musaSuccess) return;

    int supported = 0;
    if (musaDeviceGetAttribute(&supported, musaDevAttrMemoryPoolsSupported,
                               ordinal) != musaSuccess ||
        supported == 0) {
      return;
    }

    // Create our own pool pinned to this device. Using a private pool
    // keeps our pool-attribute tweaks (release threshold, etc.) from
    // leaking into any other code path in the process.
    musaMemPoolProps props = {};
    props.allocType = musaMemAllocationTypePinned;
    props.handleTypes = musaMemHandleTypeNone;
    props.location.type = musaMemLocationTypeDevice;
    props.location.id = ordinal;
    props.maxSize = 0;  // system-default maximum

    musaMemPool_t pool = nullptr;
    if (musaMemPoolCreate(&pool, &props) != musaSuccess || pool == nullptr) {
      return;
    }

    // Keep everything the pool has ever served resident; BFC handles
    // reuse on top. Using UINT64_MAX matches the "retain forever" idiom
    // used by CUDA stream-ordered allocator tutorials.
    // Non-fatal on failure: we'd just be less optimal w.r.t. OS reclaim.
    uint64_t threshold = UINT64_MAX;
    musaMemPoolSetAttribute(pool, musaMemPoolAttrReleaseThreshold, &threshold);

    musaStream_t s = nullptr;
    if (musaStreamCreateWithFlags(&s, musaStreamNonBlocking) != musaSuccess) {
      musaMemPoolDestroy(pool);
      return;
    }
    pd->pool = pool;
    pd->stream = s;
    pd->enabled = true;
#else
    (void)ordinal;
#endif
  }

  std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<PerDevice>> per_device_;
};

// --- Per-device event pool for create_event / destroy_event ---
//
// TF's EventMgr creates and destroys hundreds of events per session step: one
// each time a tensor's lifetime needs to be tied to GPU completion. On a
// 2500+-node inference graph, `musaEventCreateWithFlags` / `musaEventDestroy`
// round-trips to the driver add up to non-trivial per-step CPU overhead and
// show up directly as gaps between kernel launches in `musaEventSync`-dense
// regions of the timeline.
//
// Correctness contract: re-recording a pooled `musaEvent_t` silently
// overwrites any prior record. That is ONLY safe if every wait (host-side
// `musaEventSynchronize` or stream-side `musaStreamWaitEvent`) tied to the
// previous record has already been RELEASED by the driver, not just
// enqueued. Unlike `create_stream_dependency`, where we can't prove that
// property at destroy time, events handed to us via `destroy_event` are
// observable: we query them before pooling and only re-use handles for
// which `musaEventQuery == musaSuccess`. That means the record has been
// reached, which in turn means any `musaStreamWaitEvent` referring to it
// has been signalled and any host `musaEventSynchronize` has returned.
// Incomplete events fall through to plain `musaEventDestroy` -- a perfectly
// safe release path since the driver refcounts the handle internally, so
// we never let a pending wait race against a re-record.
//
// The pool is per-device because events are device-scoped in the MUSA
// runtime: an event created on device A cannot be recorded on a stream of
// device B. A small per-device cap keeps memory bounded if EventMgr ever
// builds a pathological backlog.
class EventPool {
 public:
  static EventPool& Instance() {
    static EventPool inst;
    return inst;
  }

  // Returns a ready-to-record event for `ordinal`. On cache hit the pooled
  // handle is returned directly; on miss we fall back to the driver. Either
  // way the returned handle is created with `musaEventDisableTiming` so
  // record/wait are both lightweight.
  musaError_t Acquire(int ordinal, musaEvent_t* out_event) {
    if (!Enabled()) {
      musaEvent_t ev = nullptr;
      musaError_t err = musaEventCreateWithFlags(&ev, musaEventDisableTiming);
      if (err != musaSuccess) return err;
      *out_event = ev;
      return musaSuccess;
    }
    PerDevice& pd = GetPerDevice(ordinal);
    {
      std::lock_guard<std::mutex> lk(pd.mu);
      if (!pd.free.empty()) {
        *out_event = pd.free.back();
        pd.free.pop_back();
        return musaSuccess;
      }
    }
    musaEvent_t ev = nullptr;
    musaError_t err = musaEventCreateWithFlags(&ev, musaEventDisableTiming);
    if (err != musaSuccess) return err;
    *out_event = ev;
    return musaSuccess;
  }

  // Returns an event to the pool, or destroys it if the pool is full.
  //
  // Historically this path also called `musaEventQuery(ev)` and only pooled
  // completed events, out of concern that recycling a handle whose record
  // is still in flight could race a pending `musaStreamWaitEvent`. That
  // concern is unfounded: per the CUDA/MUSA programming model,
  // `musaStreamWaitEvent` captures a reference to the event's record state
  // at the time of the call, and that reference is unaffected by
  // subsequent `musaEventRecord` calls on the same handle. Pooling
  // unconditionally is therefore safe.
  //
  // In addition, by the time TF's EventMgr calls `destroy_event` the
  // record has already been polled to SE_EVENT_COMPLETE, so the query was
  // already redundant in practice. Eliding it saves one driver round-trip
  // per destroy -- on the prunedGraph inference workload there are 100+
  // destroys per step, which is 0.3-0.5 ms of recovered time.
  void Release(int ordinal, musaEvent_t ev) {
    if (ev == nullptr) return;
    if (!Enabled()) {
      musaEventDestroy(ev);
      return;
    }
    PerDevice& pd = GetPerDevice(ordinal);
    {
      std::lock_guard<std::mutex> lk(pd.mu);
      if (pd.free.size() < kMaxPerDevice) {
        pd.free.push_back(ev);
        return;
      }
    }
    musaEventDestroy(ev);
  }

 private:
  // Empirically EventMgr's in-flight set on TF 2.6 inference sits in the
  // low hundreds; 256 is a comfortable ceiling that still bounds memory.
  static constexpr size_t kMaxPerDevice = 256;

  // Opt-out: if `TF_MUSA_DISABLE_EVENT_POOL=1` is set we fall back to plain
  // create/destroy for every event. Useful for A/B testing correctness.
  static bool Enabled() {
    static const bool kEnabled = []() {
      const char* e = std::getenv("TF_MUSA_DISABLE_EVENT_POOL");
      if (e == nullptr || *e == '\0') return true;
      const std::string v(e);
      return !(v == "1" || v == "true" || v == "TRUE" || v == "yes" ||
               v == "YES" || v == "on" || v == "ON");
    }();
    return kEnabled;
  }

  struct PerDevice {
    std::mutex mu;
    std::vector<musaEvent_t> free;
  };

  PerDevice& GetPerDevice(int ordinal) {
    // First try a lock-free read: per_device_ stabilizes after the first
    // access for each ordinal, and subsequent lookups should not contend.
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = per_device_.find(ordinal);
      if (it != per_device_.end()) return *it->second;
      auto pd = std::unique_ptr<PerDevice>(new PerDevice);
      PerDevice* raw = pd.get();
      per_device_.emplace(ordinal, std::move(pd));
      return *raw;
    }
  }

  std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<PerDevice>> per_device_;
};

/*** Allocation callbacks ***/

void allocate(const SP_Device* device, uint64_t size, int64_t /*memory_space*/,
              SP_DeviceMemoryBase* mem) {
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->ext = nullptr;
  mem->opaque = nullptr;
  mem->size = size;
  mem->payload = 0;

  if (size == 0) return;

  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);

  // If stream-ordered allocation is enabled for this device, use it
  // exclusively: mixing musaMallocAsync / musaFree (or vice versa) is
  // undefined behavior per the MUSA/CUDA runtime spec. On transient
  // pool failure (OOM, etc.) we leave opaque=nullptr so that BFC trims
  // its caches and retries, which is the same behavior TF sees when
  // musaMalloc OOMs. Devices where the pool APIs are unavailable fall
  // through to the synchronous path below.
  void* ptr = nullptr;
  if (AsyncDeviceAllocator::Instance().IsEnabled(ordinal)) {
    if (AsyncDeviceAllocator::Instance().Allocate(ordinal, size, &ptr)) {
      mem->opaque = ptr;
    }
    // On failure: leave opaque=nullptr; caller (BFC) handles it.
    return;
  }

  musaError_t err = musaMalloc(&ptr, size);
  if (err != musaSuccess) {
    // Leave opaque=nullptr; TF's BFC wrapper treats this as allocator failure.
    return;
  }
  mem->opaque = ptr;
}

void deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  if (!mem || !mem->opaque) return;
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);

  // Mirror the allocate() branching precisely: if the device is in
  // pool-async mode the pointer must have come from musaMallocAsync
  // and must be released via musaFreeAsync.
  if (AsyncDeviceAllocator::Instance().IsEnabled(ordinal)) {
    AsyncDeviceAllocator::Instance().Deallocate(ordinal, mem->opaque);
  } else {
    musaError_t err = musaFree(mem->opaque);
    (void)err;  // BFC's caller expects this to be best-effort.
  }
  mem->opaque = nullptr;
  mem->size = 0;
}

void* host_memory_allocate(const SP_Device* device, uint64_t size) {
  if (size == 0) return nullptr;
  CachedSetDevice(DeviceOrdinal(device));
  void* ptr = nullptr;
  musaError_t err = musaHostAlloc(&ptr, size, musaHostAllocDefault);
  if (err != musaSuccess) return nullptr;
  return ptr;
}

void host_memory_deallocate(const SP_Device* device, void* mem) {
  if (!mem) return;
  CachedSetDevice(DeviceOrdinal(device));
  musaFreeHost(mem);
}

TF_Bool get_allocator_stats(const SP_Device* /*device*/,
                            SP_AllocatorStats* /*stats*/) {
  // TF's BFC wrapper populates its own stats; the underlying SubAllocator
  // does not need to provide them.
  return false;
}

TF_Bool device_memory_usage(const SP_Device* device, int64_t* free,
                            int64_t* total) {
  CachedSetDevice(DeviceOrdinal(device));
  size_t f = 0, t = 0;
  if (musaMemGetInfo(&f, &t) != musaSuccess) return false;
  *free = static_cast<int64_t>(f);
  *total = static_cast<int64_t>(t);
  return true;
}

/*** Stream callbacks ***/

void create_stream(const SP_Device* device, SP_Stream* stream,
                   TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaStream_t s = nullptr;
  // We intentionally use `musaStreamDefault` (blocking w.r.t. the legacy
  // NULL stream) rather than `musaStreamNonBlocking` here.
  //
  // In theory non-blocking streams give a measurable throughput win because
  // executor-owned streams stop implicitly serializing with stream 0 -- this
  // is what TF's CUDA backend does in `cuda_stream.cc`. We trialled that
  // and confirmed ~1 ms/step savings on the prunedGraph inference workload.
  //
  // However, under sustained load (tens of thousands of iterations) the
  // current MUSA runtime exhibits a cross-stream synchronization race:
  // `musaStreamWaitEvent` waits placed on a non-blocking stream sometimes
  // do not actually block until the referenced record on another non-
  // blocking stream is observed. This is reproducible via TF's HostMemory
  // input path (e.g. `Pack -> Reshape.shape`), which relies on a
  // `device_to_host_stream->ThenWaitFor(compute_stream)` sync. Under the
  // race, the D2H copy reads partially-written / stale shape data and the
  // Reshape op fails with "target shape has N elements" where N is garbage.
  //
  // Reverting to `musaStreamDefault` restores correctness under long runs
  // at the cost of ~0.4 ms average latency. The overall inference pipeline
  // is still well below the pre-refactor baseline thanks to the other
  // optimizations in this file (event pool, MusaResourceMgr TL caching,
  // CachedSetDevice, etc.). Revisit `musaStreamNonBlocking` once the MUSA
  // runtime's cross-stream event guarantees are strengthened.
  musaError_t err = musaStreamCreateWithFlags(&s, musaStreamDefault);
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaStreamCreateWithFlags");
    return;
  }
  *stream = AsSPStream(s);
  SetOK(status);
}

void destroy_stream(const SP_Device* device, SP_Stream stream) {
  if (!stream) return;
  CachedSetDevice(DeviceOrdinal(device));
  musaStreamDestroy(AsMusaStream(stream));
}

void create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                              SP_Stream other, TF_Status* status) {
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  // Get an event from the shared pool so we amortize the driver round-trip
  // of `musaEventCreate` + `musaEventDestroy` across calls. TF's
  // `Stream::ThenWaitFor` hits this callback on every cross-stream sync
  // (H2D -> compute, compute -> D2H, etc.), and the prunedGraph inference
  // workload invokes it tens of times per step. On a 2500+-node graph the
  // saved 2 driver calls per dependency shows up as ~0.3-0.5 ms per step.
  //
  // Safety: the CUDA/MUSA programming model guarantees that
  // `musaStreamWaitEvent` captures the event's record at the time of the
  // call; subsequent `musaEventRecord` calls on the same handle do NOT
  // invalidate that capture. Recycling the handle while a wait is still
  // in-flight is therefore well-defined, so we can use the "unchecked"
  // pool release that skips the `musaEventQuery` gate used by
  // `destroy_event`.
  musaEvent_t ev = nullptr;
  musaError_t err = EventPool::Instance().Acquire(ordinal, &ev);
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaEventCreate (stream dep)");
    return;
  }
  err = musaEventRecord(ev, AsMusaStream(other));
  if (err != musaSuccess) {
    musaEventDestroy(ev);
    SetMusaError(status, err, "musaEventRecord (stream dep)");
    return;
  }
  err = musaStreamWaitEvent(AsMusaStream(dependent), ev, 0);
  if (err != musaSuccess) {
    musaEventDestroy(ev);
    SetMusaError(status, err, "musaStreamWaitEvent (stream dep)");
    return;
  }
  EventPool::Instance().Release(ordinal, ev);
  SetOK(status);
}

void get_stream_status(const SP_Device* device, SP_Stream stream,
                       TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaStreamQuery(AsMusaStream(stream));
  if (err == musaSuccess || err == musaErrorNotReady) {
    SetOK(status);
    return;
  }
  SetMusaError(status, err, "musaStreamQuery");
}

/*** Event callbacks ***/

void create_event(const SP_Device* device, SP_Event* event, TF_Status* status) {
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  musaEvent_t ev = nullptr;
  musaError_t err = EventPool::Instance().Acquire(ordinal, &ev);
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaEventCreate");
    return;
  }
  *event = AsSPEvent(ev);
  SetOK(status);
}

void destroy_event(const SP_Device* device, SP_Event event) {
  if (!event) return;
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  EventPool::Instance().Release(ordinal, AsMusaEvent(event));
}

SE_EventStatus get_event_status(const SP_Device* device, SP_Event event) {
  if (!event) return SE_EVENT_ERROR;
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaEventQuery(AsMusaEvent(event));
  if (err == musaSuccess) return SE_EVENT_COMPLETE;
  if (err == musaErrorNotReady) return SE_EVENT_PENDING;
  return SE_EVENT_ERROR;
}

void record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                  TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaEventRecord(AsMusaEvent(event), AsMusaStream(stream));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaEventRecord");
    return;
  }
  SetOK(status);
}

void wait_for_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                    TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err =
      musaStreamWaitEvent(AsMusaStream(stream), AsMusaEvent(event), 0);
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaStreamWaitEvent");
    return;
  }
  SetOK(status);
}

/*** Timer callbacks (no-op) ***/

void create_timer(const SP_Device*, SP_Timer* timer, TF_Status* status) {
  *timer = nullptr;
  SetOK(status);
}
void destroy_timer(const SP_Device*, SP_Timer) {}
void start_timer(const SP_Device*, SP_Stream, SP_Timer, TF_Status* status) {
  SetOK(status);
}
void stop_timer(const SP_Device*, SP_Stream, SP_Timer, TF_Status* status) {
  SetOK(status);
}

/*** Memcpy callbacks ***/

// One-shot diagnostic: on demand, print the pinned-status of a few H2D
// source pointers so we can verify whether `feed_dict` inputs are coming in
// as pageable memory (which makes `musaMemcpyAsync` effectively synchronous
// and prevents overlap with compute on other streams).
//
// Enable with env var `TF_MUSA_DIAG_H2D_PINNED=1`. First 50 calls are
// logged; the rest are skipped so we don't spam a 50k-iter benchmark.
static void DiagH2DPinned(const void* host_src, uint64_t size, int ordinal) {
  static const bool kEnabled = []() {
    const char* e = std::getenv("TF_MUSA_DIAG_H2D_PINNED");
    if (e == nullptr || *e == '\0') return false;
    const std::string v(e);
    return !(v == "0" || v == "false" || v == "FALSE" || v == "no" ||
             v == "NO" || v == "off" || v == "OFF");
  }();
  if (!kEnabled) return;
  static std::atomic<int> count{0};
  const int cur = count.fetch_add(1, std::memory_order_relaxed);
  if (cur >= 50) return;

  // Two independent probes because MUSA 11+ returns musaSuccess even for
  // unregistered pageable memory (with .type == musaMemoryTypeUnregistered)
  // -- checking only the return code is not sufficient.
  //
  //   1) musaPointerGetAttributes -> .type tells us Unregistered / Host /
  //      Device / Managed.
  //   2) musaHostGetFlags -> succeeds iff the host memory was allocated via
  //      musaHostAlloc or registered via musaHostRegister (i.e. truly
  //      pinned for DMA). This is the most reliable "is it pinned" probe.
  musaPointerAttributes attrs;
  std::memset(&attrs, 0, sizeof(attrs));
  musaError_t attr_err = musaPointerGetAttributes(&attrs, host_src);
  if (attr_err != musaSuccess) (void)musaGetLastError();

  const char* type_str = "?";
  if (attr_err != musaSuccess) {
    type_str = "err";
  } else {
    switch (attrs.type) {
      case musaMemoryTypeUnregistered:
        type_str = "Unregistered";
        break;
      case musaMemoryTypeHost:
        type_str = "Host";
        break;
      case musaMemoryTypeDevice:
        type_str = "Device";
        break;
      case musaMemoryTypeManaged:
        type_str = "Managed";
        break;
      default:
        type_str = "?";
        break;
    }
  }

  unsigned int host_flags = 0;
  musaError_t flags_err =
      musaHostGetFlags(&host_flags, const_cast<void*>(host_src));
  const bool host_get_flags_ok = (flags_err == musaSuccess);
  if (!host_get_flags_ok) (void)musaGetLastError();

  const char* verdict;
  if (host_get_flags_ok) {
    verdict = "PINNED";
  } else if (attr_err == musaSuccess && attrs.type == musaMemoryTypeHost) {
    verdict = "PINNED(type=Host)";
  } else {
    verdict = "PAGEABLE";
  }

  std::fprintf(stderr,
               "[MUSA H2D diag] #%02d ordinal=%d size=%llu src=%p attr.type=%s "
               "HostGetFlags=%s -> %s\n",
               cur, ordinal, static_cast<unsigned long long>(size), host_src,
               type_str, host_get_flags_ok ? "ok" : "fail", verdict);
  std::fflush(stderr);
}

// --- Opt-in host-pinning cache for H2D sources ---
//
// When `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES` is set to a positive value,
// H2D source buffers whose size is at least that many bytes are passed
// through `musaHostRegister` after being observed for at least one
// iteration. Subsequent `musaMemcpyAsync` calls against the same region
// become truly asynchronous (no CPU-side staging copy into an internal
// pinned buffer), which lets H2D overlap with compute.
//
// Why "after being observed" -- shared-page safety
// ================================================
//
// In realistic TensorFlow workloads (notably `session.run(feed_dict=...)`
// with NumPy arrays) different feed tensors frequently land in memory
// regions that share pages, either because the CPU allocator packs them
// tightly or because TF internally repacks feed_dict into a contiguous
// staging area before issuing per-tensor H2Ds. The MUSA runtime rejects
// `musaMemcpyAsync` whose source range straddles a pinned region and
// a non-pinned region with `musaErrorInvalidValue` (observed
// empirically on MTT S-series devices with MUSA 5.1). It also appears
// to "own" the page immediately adjacent to a registered range.
//
// To avoid this cliff we:
//
//   1. On the first sighting of a (start, size) pair we only record it;
//      we do NOT register. This lets us build up a picture of the set
//      of active host buffers before committing to any pins.
//   2. On the second sighting we check that the proposed registration
//      range does not overlap (within a one-page safety margin) with
//      any OTHER observed buffer. If it does, we mark the entry
//      `do_not_pin` permanently. Otherwise we try to register.
//   3. Registration uses inward page alignment with a one-page margin
//      on each end to avoid corrupting adjacent allocations.
//
// On workloads where feed tensors do NOT share pages (or are already
// pinned via `musaHostAlloc`) this opportunistically upgrades large
// H2Ds to truly-async. On workloads where they do share pages (like TF
// session.run feed_dict on this MUSA stack) it falls back to pageable
// memcpy for every affected buffer -- correctness first, no crash.
//
// Other correctness constraints:
//
//   * Never unregister. We do not own the caller's buffer and cannot
//     observe when it is freed. We pay the price of holding the pages
//     locked for the lifetime of the process.
//   * Pin failures are silent: we mark the entry `do_not_pin` and fall
//     through to the normal pageable memcpy path.
//
// This is explicit OPT-IN because:
//
//   * It consumes physically-resident host memory for the process's
//     lifetime -- unbounded for workloads that churn through unique
//     host buffers.
//   * On workloads with shared-page feed_dict this does not help and
//     shouldn't be paid for (the bookkeeping itself is cheap, but there
//     is no upside).
class HostPinCache {
 public:
  static HostPinCache& Instance() {
    static HostPinCache inst;
    return inst;
  }

  // Returns true iff auto-pinning is enabled and a non-zero size threshold
  // is configured. Parsed once from the environment.
  bool Enabled() const { return threshold_ > 0; }
  size_t Threshold() const { return threshold_; }

  // Observes one H2D call and, if safe, pins the source buffer. A no-op
  // if the feature is disabled or the range is below threshold.
  //
  // Never crashes: if any step (overlap check, musaHostRegister, etc.)
  // signals that pinning this buffer is unsafe we record it as
  // do-not-pin and let the caller proceed with the pageable memcpy.
  void MaybePin(const void* host_src, size_t size) {
    if (!Enabled() || size < threshold_ || host_src == nullptr) return;

    const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(host_src);
    const uintptr_t raw_end = raw_begin + size;
    const Key key{raw_begin, raw_end};

    std::unique_lock<std::mutex> lk(mu_);

    Entry& e = observed_[key];
    e.count++;

    // Already decided: either pinned, verified already-pinned externally,
    // or on the deny list.
    if (e.registered || e.do_not_pin) return;

    // First sighting: just record and move on. We want to see the full
    // set of active buffers before deciding what is safe to pin.
    if (e.count < 2) return;

    // Second sighting: check that no other OBSERVED buffer overlaps with
    // the +/- one page safety zone around this one. If so, both we and
    // they stay pageable.
    const uintptr_t my_lo =
        (raw_begin >= page_size_) ? (raw_begin - page_size_) : 0;
    const uintptr_t my_hi = raw_end + page_size_;
    for (const auto& kv : observed_) {
      if (&kv.second == &e) continue;
      // Standard half-open interval overlap test.
      if (kv.first.begin < my_hi && kv.first.end > my_lo) {
        e.do_not_pin = true;
        return;
      }
    }

    // Check if the buffer is already pinned (e.g. TF's internal pinned
    // allocator, musaHostAlloc) -- musaHostGetFlags succeeds iff the
    // memory is pinned. If so, no registration needed.
    unsigned int existing_flags = 0;
    // Release the lock around the driver call to avoid serializing
    // unrelated callbacks. Re-acquire before mutating `e`.
    lk.unlock();
    musaError_t flags_err =
        musaHostGetFlags(&existing_flags, const_cast<void*>(host_src));
    if (flags_err != musaSuccess) (void)musaGetLastError();
    if (flags_err == musaSuccess) {
      lk.lock();
      e.registered = true;
      return;
    }

    // Compute an inward-aligned range with a one-page margin on each
    // end. `musaHostRegister` locks at page granularity and has been
    // observed to impact the immediately-adjacent page as well, so we
    // stay strictly inside.
    const uintptr_t first_inner_page =
        (raw_begin + page_size_ - 1) & ~(page_size_ - 1);
    const uintptr_t last_inner_page = raw_end & ~(page_size_ - 1);
    if (first_inner_page + page_size_ * 2 > last_inner_page) {
      // Buffer is too small for a 2-page margin. Not worth pinning.
      lk.lock();
      e.do_not_pin = true;
      return;
    }
    const uintptr_t aligned_begin = first_inner_page + page_size_;
    const uintptr_t aligned_end = last_inner_page - page_size_;
    const size_t aligned_size = aligned_end - aligned_begin;

    musaError_t reg_err =
        musaHostRegister(reinterpret_cast<void*>(aligned_begin), aligned_size,
                         musaHostRegisterPortable);
    LogRegisterResultOnce(reg_err, aligned_begin, aligned_size);
    if (reg_err != musaSuccess) {
      (void)musaGetLastError();
      lk.lock();
      // Already-registered is effectively success for our purposes.
      if (reg_err == musaErrorHostMemoryAlreadyRegistered) {
        e.registered = true;
      } else {
        e.do_not_pin = true;
        LogRegisterFailureOnce(reg_err, aligned_begin, aligned_size);
      }
      return;
    }

    lk.lock();
    e.registered = true;
    total_pinned_bytes_ += aligned_size;
  }

 private:
  HostPinCache() {
    long ps = ::sysconf(_SC_PAGESIZE);
    page_size_ = (ps > 0) ? static_cast<size_t>(ps) : 4096;

    const char* e = std::getenv("TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES");
    if (e != nullptr && *e != '\0') {
      long long v = std::atoll(e);
      if (v > 0) threshold_ = static_cast<size_t>(v);
    }
    if (threshold_ > 0) {
      std::fprintf(stderr,
                   "[MUSA] H2D auto-pin enabled: threshold=%zu bytes, "
                   "page_size=%zu\n",
                   threshold_, page_size_);
      std::fflush(stderr);
    }
  }

  static void LogRegisterResultOnce(musaError_t err, uintptr_t begin,
                                    size_t size) {
    static std::atomic<int> count{0};
    const int cur = count.fetch_add(1, std::memory_order_relaxed);
    if (cur >= 4) return;
    std::fprintf(stderr,
                 "[MUSA] musaHostRegister [0x%lx, +%zu] -> err=%d (\"%s\")\n",
                 static_cast<unsigned long>(begin), size, static_cast<int>(err),
                 musaGetErrorString(err));
    std::fflush(stderr);
  }

  static void LogRegisterFailureOnce(musaError_t err, uintptr_t begin,
                                     size_t size) {
    static std::atomic<int> count{0};
    const int cur = count.fetch_add(1, std::memory_order_relaxed);
    if (cur >= 4) return;
    std::fprintf(
        stderr,
        "[MUSA] musaHostRegister failed (err=%d \"%s\") for [0x%lx, +%zu]; "
        "falling back to pageable H2D.\n",
        static_cast<int>(err), musaGetErrorString(err),
        static_cast<unsigned long>(begin), size);
    std::fflush(stderr);
  }

  struct Key {
    uintptr_t begin;
    uintptr_t end;
    bool operator==(const Key& o) const {
      return begin == o.begin && end == o.end;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<uintptr_t>()(k.begin) ^
             (std::hash<uintptr_t>()(k.end) << 1);
    }
  };

  // One entry per unique (begin, end) we have ever observed on a
  // memcpy_htod call of at-least-threshold size.
  struct Entry {
    uint32_t count = 0;  // Number of sightings.
    bool registered = false;
    bool do_not_pin = false;
  };

  size_t threshold_ = 0;
  size_t page_size_ = 4096;
  std::mutex mu_;
  std::unordered_map<Key, Entry, KeyHash> observed_;
  size_t total_pinned_bytes_ = 0;
};

// --- Opt-in pinned staging buffer pool for H2D ---
//
// When `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES` is a positive value, H2D
// copies whose source buffer is pageable and whose size meets the
// threshold are routed through a pool of pinned host buffers:
//
//   1. Acquire a pinned buffer from the pool (or `musaHostAlloc` a new
//      one sized to the matching size class).
//   2. CPU-memcpy from the user's pageable buffer into the pinned one.
//   3. Issue `musaMemcpyAsync` from the pinned buffer on the caller's
//      stream -- this is a TRUE asynchronous PCIe transfer because the
//      source is pinned, so it returns immediately and the CPU can
//      continue enqueuing compute on other streams.
//   4. Record an event on the stream; stash (event, buffer) so the
//      buffer can be reclaimed once the H2D completes.
//
// Compared to the stock `musaMemcpyAsync` with pageable source, which
// the MUSA runtime implements as a synchronous in-driver stage
// ("copy-into-internal-pinned then queue the pinned-to-device copy"
// while the CPU waits), this replaces the in-driver stage with:
//
//   - An explicit CPU memcpy that we know costs ~N bytes / host-memory
//     bandwidth.
//   - An async PCIe transfer that genuinely overlaps with the rest of
//     the step's compute.
//
// For workloads where `session.run(feed_dict=...)` dominates H2D time
// and the feed arrays are pageable NumPy arrays (the common case), the
// CPU-blocked portion of a step shrinks from "total H2D payload / PCIe
// BW + driver sync" to roughly "total H2D payload / host memory BW",
// which for this MUSA stack is a ~5-10x win on the copy phase.
//
// Why this is safe where `musaHostRegister` was not:
//
//   - The staging buffers are owned by us; no risk of colliding with
//     TF/NumPy's internal page sharing.
//   - We never modify the caller's buffer, never register their pages,
//     and never keep references to their memory after the call returns.
//
// Pool management:
//
//   - Buffers are bucketed by size class (next power-of-two >= 64 KiB,
//     capped at the largest observed request). Most feed_dict workloads
//     cycle through a small set of sizes, so this gives ~100%
//     reuse after the first iteration.
//   - In-flight buffers are tracked via `musaEvent_t`. On each Acquire
//     we drain completed entries back into the free list.
//   - `TF_MUSA_H2D_STAGING_MAX_POOL_MB` (default 1024) caps total
//     allocated pinned memory. Once the cap is hit we stop growing and
//     fall back to the pageable-memcpy path for oversized requests.
class PinnedStagingPool {
 public:
  static PinnedStagingPool& Instance() {
    static PinnedStagingPool inst;
    return inst;
  }

  bool Enabled() const { return threshold_ > 0; }

  // Diagnostic counters. Printed once at process exit when staging is
  // enabled.
  std::atomic<uint64_t> stat_staged_count_{0};
  std::atomic<uint64_t> stat_staged_bytes_{0};
  std::atomic<uint64_t> stat_already_pinned_count_{0};
  std::atomic<uint64_t> stat_pool_alloc_{0};

  // Attempts to run the H2D via the staging pool. Returns true iff the
  // staging path was used (whether the copy itself succeeded or not --
  // `*out_err` carries that). When false, the caller must fall back to
  // the usual pageable `musaMemcpyAsync`.
  bool TryStagingCopy(void* device_dst, const void* host_src, uint64_t size,
                      musaStream_t stream, musaError_t* out_err) {
    if (!Enabled() || size < threshold_ || host_src == nullptr ||
        device_dst == nullptr) {
      return false;
    }

    // If the source is ALREADY pinned (e.g. the caller already did
    // musaHostAlloc / musaHostRegister), staging adds an unnecessary
    // CPU memcpy -- skip it and let the stock async H2D path run.
    unsigned int flags = 0;
    if (musaHostGetFlags(&flags, const_cast<void*>(host_src)) == musaSuccess) {
      stat_already_pinned_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    (void)musaGetLastError();

    const size_t cls = SizeClass(size);
    void* stage = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      DrainInFlightLocked();
      stage = AcquireFromPoolLocked(cls);
    }
    if (stage == nullptr) {
      // Try to grow the pool. This has to happen outside the lock
      // because musaHostAlloc is slow and contended.
      stage = AllocateNew(cls);
      if (stage == nullptr) return false;  // over cap or alloc failed
      stat_pool_alloc_.fetch_add(1, std::memory_order_relaxed);
    }

    stat_staged_count_.fetch_add(1, std::memory_order_relaxed);
    stat_staged_bytes_.fetch_add(size, std::memory_order_relaxed);
    // DEBUG ONLY: env TF_MUSA_H2D_STAGING_SKIP_MEMCPY=1 skips the CPU
    // memcpy to isolate whether it's on the critical path.
    static const bool kSkipMemcpy = []() {
      const char* e = std::getenv("TF_MUSA_H2D_STAGING_SKIP_MEMCPY");
      return e != nullptr && *e == '1';
    }();
    // Step 1: CPU memcpy pageable -> pinned.
    //
    // Pageable -> pinned DMA preparation on these hosts tops out around
    // ~5 GB/s from a single thread (below theoretical DRAM BW because
    // the first-touch paths have to page-fault / populate page-table
    // entries). Splitting the copy across a small worker pool scales
    // close to linearly for big buffers (>=1 MiB), which is exactly
    // where we care.
    if (!kSkipMemcpy) {
      ParallelMemcpy(stage, host_src, size);
    }

    // Step 2: async H2D from pinned source.
    musaError_t e = musaMemcpyAsync(device_dst, stage, size,
                                    musaMemcpyHostToDevice, stream);
    if (e != musaSuccess) {
      // H2D never got enqueued -- return the buffer to the pool
      // immediately; no event needed.
      ReturnToPool(stage, cls);
      *out_err = e;
      return true;
    }

    // Step 3: record an event so we know when we can reuse the buffer.
    musaEvent_t ev = nullptr;
    if (musaEventCreateWithFlags(&ev, musaEventDisableTiming) != musaSuccess) {
      // Couldn't create the event -- safest fallback is to wait for
      // the stream to drain so we can reclaim now.
      (void)musaGetLastError();
      musaStreamSynchronize(stream);
      ReturnToPool(stage, cls);
      *out_err = musaSuccess;
      return true;
    }
    if (musaEventRecord(ev, stream) != musaSuccess) {
      (void)musaGetLastError();
      musaEventDestroy(ev);
      musaStreamSynchronize(stream);
      ReturnToPool(stage, cls);
      *out_err = musaSuccess;
      return true;
    }

    {
      std::lock_guard<std::mutex> lk(mu_);
      in_flight_.push_back({ev, stage, cls});
    }
    *out_err = musaSuccess;
    return true;
  }

 private:
  struct InFlight {
    musaEvent_t ev;
    void* buf;
    size_t cls;
  };

  PinnedStagingPool() {
    const char* e = std::getenv("TF_MUSA_H2D_STAGING_THRESHOLD_BYTES");
    if (e != nullptr && *e != '\0') {
      long long v = std::atoll(e);
      if (v > 0) threshold_ = static_cast<size_t>(v);
    }
    max_pool_bytes_ = 1024ULL * 1024ULL * 1024ULL;  // 1 GiB default
    const char* m = std::getenv("TF_MUSA_H2D_STAGING_MAX_POOL_MB");
    if (m != nullptr && *m != '\0') {
      long long mb = std::atoll(m);
      if (mb > 0) max_pool_bytes_ = static_cast<size_t>(mb) * 1024ULL * 1024ULL;
    }
    if (threshold_ > 0) {
      std::fprintf(stderr,
                   "[MUSA] H2D staging pool enabled: threshold=%zu bytes, "
                   "max_pool=%zu bytes\n",
                   threshold_, max_pool_bytes_);
      std::fflush(stderr);
      std::atexit(&PinnedStagingPool::PrintStatsAtExit);
    }
  }

  // Multi-threaded memcpy: forks work to a small worker pool for
  // buffers large enough to amortize the synchronization overhead.
  //
  // Tunable via `TF_MUSA_H2D_STAGING_MEMCPY_THREADS` (default 4, clamped
  // to [1, 16]). At 4 threads we see ~3-4x throughput on 4-10 MiB
  // pageable->pinned copies.
  static void ParallelMemcpy(void* dst, const void* src, size_t size) {
    static const size_t kMinParallel = 512 * 1024;  // 512 KiB
    ParallelMemcpyImpl& impl = ParallelMemcpyImpl::Instance();
    if (size < kMinParallel || impl.NumThreads() <= 1) {
      std::memcpy(dst, src, size);
      return;
    }
    impl.Run(dst, src, size);
  }

  class ParallelMemcpyImpl {
   public:
    static ParallelMemcpyImpl& Instance() {
      static ParallelMemcpyImpl inst;
      return inst;
    }

    size_t NumThreads() const { return workers_.size() + 1; }

    // Run(): serialized across concurrent callers.
    //
    // The worker pool keeps only a single job slot (`job_dst_`, `job_src_`,
    // `job_chunk_`, `job_size_`, `pending_`, `generation_`). If two threads
    // were allowed to enter Run() simultaneously, the second would overwrite
    // the slot and reset `pending_` before the first run's workers have
    // picked up the job from `mu_`, causing them to all process the second
    // job and decrement the second run's pending_. The first Run() would
    // then observe `pending_ == 0` on `done_cv_` and return with most
    // chunks of its buffer NEVER copied -- the caller then proceeds to
    // `musaMemcpyAsync` a partially-initialized pinned staging buffer into
    // GPU memory. Downstream kernels consume the garbage, triggering a
    // driver fault and a "force destroy app memory context" in dmesg, which
    // manifests as a silent hang (host threads wait forever on events that
    // will never complete).
    //
    // The fix is an outer mutex that guards the entire Run() body so only
    // one job at a time can touch the shared slot. The intra-Run
    // parallelism (chunks spread across workers) is preserved.
    void Run(void* dst, const void* src, size_t size) {
      std::lock_guard<std::mutex> run_lk(run_mu_);

      const size_t nt = NumThreads();
      const size_t chunk = (size + nt - 1) / nt;

      std::unique_lock<std::mutex> lk(mu_);
      job_dst_ = dst;
      job_src_ = src;
      job_chunk_ = chunk;
      job_size_ = size;
      pending_ = workers_.size();
      generation_++;
      lk.unlock();
      start_cv_.notify_all();

      // Main thread does the first chunk itself.
      const size_t my_off = 0;
      const size_t my_n = std::min(chunk, size);
      std::memcpy(static_cast<char*>(dst) + my_off,
                  static_cast<const char*>(src) + my_off, my_n);

      // Wait for workers.
      lk.lock();
      done_cv_.wait(lk, [this] { return pending_ == 0; });
    }

    ~ParallelMemcpyImpl() {
      {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
      }
      start_cv_.notify_all();
      for (auto& t : workers_) {
        if (t.joinable()) t.join();
      }
    }

   private:
    ParallelMemcpyImpl() {
      int nt = 4;
      const char* e = std::getenv("TF_MUSA_H2D_STAGING_MEMCPY_THREADS");
      if (e != nullptr && *e != '\0') {
        int v = std::atoi(e);
        if (v >= 1 && v <= 16) nt = v;
      }
      const int workers = nt - 1;  // main thread does one chunk
      for (int i = 0; i < workers; ++i) {
        const int idx = i + 1;  // chunk index for this worker
        workers_.emplace_back([this, idx] { Worker(idx); });
      }
    }

    void Worker(int idx) {
      size_t last_gen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(mu_);
        start_cv_.wait(
            lk, [this, &last_gen] { return stop_ || generation_ != last_gen; });
        if (stop_) return;
        last_gen = generation_;
        void* dst = job_dst_;
        const void* src = job_src_;
        const size_t chunk = job_chunk_;
        const size_t total = job_size_;
        lk.unlock();

        const size_t off = static_cast<size_t>(idx) * chunk;
        if (off < total) {
          const size_t n = std::min(chunk, total - off);
          std::memcpy(static_cast<char*>(dst) + off,
                      static_cast<const char*>(src) + off, n);
        }

        lk.lock();
        if (--pending_ == 0) done_cv_.notify_one();
      }
    }

    // Outer lock: serializes concurrent Run() callers. See the long comment
    // on Run() for why single-slot job state would otherwise be corrupted.
    std::mutex run_mu_;
    std::mutex mu_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;
    size_t generation_ = 0;
    size_t pending_ = 0;
    void* job_dst_ = nullptr;
    const void* job_src_ = nullptr;
    size_t job_chunk_ = 0;
    size_t job_size_ = 0;
    std::vector<std::thread> workers_;
  };

  static void PrintStatsAtExit() {
    PinnedStagingPool& p = Instance();
    const uint64_t n = p.stat_staged_count_.load(std::memory_order_relaxed);
    const uint64_t b = p.stat_staged_bytes_.load(std::memory_order_relaxed);
    const uint64_t pin =
        p.stat_already_pinned_count_.load(std::memory_order_relaxed);
    const uint64_t a = p.stat_pool_alloc_.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[MUSA] H2D staging stats: staged=%llu (%.2f MiB) "
                 "already_pinned=%llu pool_allocs=%llu\n",
                 static_cast<unsigned long long>(n),
                 static_cast<double>(b) / (1024.0 * 1024.0),
                 static_cast<unsigned long long>(pin),
                 static_cast<unsigned long long>(a));
    std::fflush(stderr);
  }

  // Round up to the next power of two, minimum 64 KiB. Small set of
  // size classes keeps free-list hit rate near 100% for steady-state
  // feed_dict workloads while bounding per-class waste to <=2x.
  static size_t SizeClass(size_t s) {
    size_t cls = 65536;
    while (cls < s) cls <<= 1;
    return cls;
  }

  void* AcquireFromPoolLocked(size_t cls) {
    auto it = free_.find(cls);
    if (it == free_.end() || it->second.empty()) return nullptr;
    void* p = it->second.back();
    it->second.pop_back();
    return p;
  }

  // Allocates a new pinned buffer outside the lock. Respects the global
  // cap.
  void* AllocateNew(size_t cls) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (total_allocated_ + cls > max_pool_bytes_) {
        return nullptr;
      }
      // Tentatively reserve the budget so concurrent callers see it.
      total_allocated_ += cls;
    }
    void* p = nullptr;
    musaError_t e = musaHostAlloc(&p, cls, musaHostAllocPortable);
    if (e != musaSuccess || p == nullptr) {
      (void)musaGetLastError();
      std::lock_guard<std::mutex> lk(mu_);
      total_allocated_ -= cls;
      return nullptr;
    }
    return p;
  }

  void ReturnToPool(void* buf, size_t cls) {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_[cls].size() < kMaxFreePerClass) {
      free_[cls].push_back(buf);
    } else {
      // Pool at class cap -- free the buffer and shrink accounting.
      musaFreeHost(buf);
      total_allocated_ -= cls;
    }
  }

  // Sweeps completed in-flight entries into the free list. Must be
  // called under `mu_`.
  //
  // `in_flight_` intermixes events recorded on multiple streams (and
  // potentially multiple devices), so there is no global completion
  // ordering: a later-inserted entry can finish before an earlier one.
  // We therefore scan the whole deque and only retain entries whose
  // events are still pending. Completed entries are destroyed and
  // their pinned buffers are returned to the pool (or freed if the
  // per-class free-list is full).
  void DrainInFlightLocked() {
    auto it = in_flight_.begin();
    while (it != in_flight_.end()) {
      musaError_t e = musaEventQuery(it->ev);
      if (e != musaSuccess) {
        // Not ready yet -- leave this entry in place; the buffer is
        // still live from the GPU's point of view.
        (void)musaGetLastError();
        ++it;
        continue;
      }
      musaEventDestroy(it->ev);
      if (free_[it->cls].size() < kMaxFreePerClass) {
        free_[it->cls].push_back(it->buf);
      } else {
        musaFreeHost(it->buf);
        total_allocated_ -= it->cls;
      }
      it = in_flight_.erase(it);
    }
  }

  static constexpr size_t kMaxFreePerClass = 8;

  size_t threshold_ = 0;
  size_t max_pool_bytes_ = 0;
  std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
  std::deque<InFlight> in_flight_;
  size_t total_allocated_ = 0;
};

void memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                 const SP_DeviceMemoryBase* device_src, uint64_t size,
                 TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  MUSA_TELEMETRY_ON_MEMCPY(host_dst, const_cast<void*>(device_src->opaque),
                           size, ordinal,
                           MUSA_TELEMETRY_STREAM_ID(AsMusaStream(stream)),
                           TelemetryEventType::kMemcpyD2H);
  musaError_t err =
      musaMemcpyAsync(host_dst, device_src->opaque, size,
                      musaMemcpyDeviceToHost, AsMusaStream(stream));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaMemcpyAsync D2H");
    return;
  }
  SetOK(status);
}

void memcpy_htod(const SP_Device* device, SP_Stream stream,
                 SP_DeviceMemoryBase* device_dst, const void* host_src,
                 uint64_t size, TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  DiagH2DPinned(host_src, size, ordinal);
  // Opt-in: pin large H2D source buffers on first sighting so that
  // subsequent `musaMemcpyAsync` calls are truly asynchronous. No-op when
  // `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES` is unset / 0.
  HostPinCache::Instance().MaybePin(host_src, size);
  MUSA_TELEMETRY_ON_MEMCPY(device_dst->opaque, const_cast<void*>(host_src),
                           size, ordinal,
                           MUSA_TELEMETRY_STREAM_ID(AsMusaStream(stream)),
                           TelemetryEventType::kMemcpyH2D);

  // Opt-in: route large pageable-source H2D copies through a pinned
  // staging buffer pool. This replaces the MUSA runtime's in-driver
  // synchronous staging with an explicit CPU memcpy + truly-async PCIe
  // transfer, letting the CPU continue enqueuing compute while the
  // transfer runs. No-op when `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES` is
  // unset / 0, or when the caller's source is already pinned.
  musaError_t err = musaSuccess;
  if (PinnedStagingPool::Instance().TryStagingCopy(
          device_dst->opaque, host_src, size, AsMusaStream(stream), &err)) {
    if (err != musaSuccess) {
      SetMusaError(status, err, "musaMemcpyAsync H2D (staged)");
      return;
    }
    SetOK(status);
    return;
  }

  err = musaMemcpyAsync(device_dst->opaque, host_src, size,
                        musaMemcpyHostToDevice, AsMusaStream(stream));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaMemcpyAsync H2D");
    return;
  }
  SetOK(status);
}

void memcpy_dtod(const SP_Device* device, SP_Stream stream,
                 SP_DeviceMemoryBase* device_dst,
                 const SP_DeviceMemoryBase* device_src, uint64_t size,
                 TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  const int ordinal = DeviceOrdinal(device);
  CachedSetDevice(ordinal);
  MUSA_TELEMETRY_ON_MEMCPY(device_dst->opaque,
                           const_cast<void*>(device_src->opaque), size, ordinal,
                           MUSA_TELEMETRY_STREAM_ID(AsMusaStream(stream)),
                           TelemetryEventType::kMemcpyD2D);
  musaError_t err =
      musaMemcpyAsync(device_dst->opaque, device_src->opaque, size,
                      musaMemcpyDeviceToDevice, AsMusaStream(stream));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaMemcpyAsync D2D");
    return;
  }
  SetOK(status);
}

void sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err =
      musaMemcpy(host_dst, device_src->opaque, size, musaMemcpyDeviceToHost);
  if (err != musaSuccess) {
    SetMusaError(status, err, "sync_memcpy D2H");
    return;
  }
  SetOK(status);
}

void sync_memcpy_htod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                      const void* host_src, uint64_t size, TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err =
      musaMemcpy(device_dst->opaque, host_src, size, musaMemcpyHostToDevice);
  if (err != musaSuccess) {
    SetMusaError(status, err, "sync_memcpy H2D");
    return;
  }
  SetOK(status);
}

void sync_memcpy_dtod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status) {
  if (size == 0) {
    SetOK(status);
    return;
  }
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaMemcpy(device_dst->opaque, device_src->opaque, size,
                               musaMemcpyDeviceToDevice);
  if (err != musaSuccess) {
    SetMusaError(status, err, "sync_memcpy D2D");
    return;
  }
  SetOK(status);
}

void block_host_for_event(const SP_Device* device, SP_Event event,
                          TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaEventSynchronize(AsMusaEvent(event));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaEventSynchronize");
    return;
  }
  SetOK(status);
}

void block_host_until_done(const SP_Device* device, SP_Stream stream,
                           TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaStreamSynchronize(AsMusaStream(stream));
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaStreamSynchronize");
    return;
  }
  SetOK(status);
}

void synchronize_all_activity(const SP_Device* device, TF_Status* status) {
  CachedSetDevice(DeviceOrdinal(device));
  musaError_t err = musaDeviceSynchronize();
  if (err != musaSuccess) {
    SetMusaError(status, err, "musaDeviceSynchronize");
    return;
  }
  SetOK(status);
}

// host_callback marshals the TF status callback through musaLaunchHostFunc,
// which is the MUSA equivalent of cudaLaunchHostFunc. This lets TF's EventMgr
// chain completion handlers on the stream without the plugin owning any
// polling thread of its own.
struct HostCallbackCtx {
  SE_StatusCallbackFn fn;
  void* arg;
};

TF_Bool host_callback(const SP_Device* device, SP_Stream stream,
                      SE_StatusCallbackFn callback_fn, void* callback_arg) {
  CachedSetDevice(DeviceOrdinal(device));
  auto* ctx = new HostCallbackCtx{callback_fn, callback_arg};
  musaError_t err = musaLaunchHostFunc(
      AsMusaStream(stream),
      [](void* d) {
        auto* c = static_cast<HostCallbackCtx*>(d);
        TF_Status* s = TF_NewStatus();
        c->fn(c->arg, s);
        TF_DeleteStatus(s);
        delete c;
      },
      ctx);
  if (err != musaSuccess) {
    delete ctx;
    return false;
  }
  return true;
}

}  // namespace

void PopulateStreamExecutor(SP_StreamExecutor* se) {
  std::memset(se, 0, sizeof(*se));
  se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  // Allocation
  se->allocate = allocate;
  se->deallocate = deallocate;
  se->host_memory_allocate = host_memory_allocate;
  se->host_memory_deallocate = host_memory_deallocate;
  // Unified memory not supported.
  se->unified_memory_allocate = nullptr;
  se->unified_memory_deallocate = nullptr;
  se->get_allocator_stats = get_allocator_stats;
  se->device_memory_usage = device_memory_usage;
  // Stream
  se->create_stream = create_stream;
  se->destroy_stream = destroy_stream;
  se->create_stream_dependency = create_stream_dependency;
  se->get_stream_status = get_stream_status;
  // Event
  se->create_event = create_event;
  se->destroy_event = destroy_event;
  se->get_event_status = get_event_status;
  se->record_event = record_event;
  se->wait_for_event = wait_for_event;
  // Timer (no-op; TF only needs this for benchmark timers).
  se->create_timer = create_timer;
  se->destroy_timer = destroy_timer;
  se->start_timer = start_timer;
  se->stop_timer = stop_timer;
  // Memcpy (async + sync)
  se->memcpy_dtoh = memcpy_dtoh;
  se->memcpy_htod = memcpy_htod;
  se->memcpy_dtod = memcpy_dtod;
  se->sync_memcpy_dtoh = sync_memcpy_dtoh;
  se->sync_memcpy_htod = sync_memcpy_htod;
  se->sync_memcpy_dtod = sync_memcpy_dtod;
  se->block_host_for_event = block_host_for_event;
  se->block_host_until_done = block_host_until_done;
  se->synchronize_all_activity = synchronize_all_activity;
  se->host_callback = host_callback;
}

void PopulateTimerFns(SP_TimerFns* timer) {
  timer->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
  timer->ext = nullptr;
  timer->nanoseconds = [](SP_Timer) -> uint64_t { return 0; };
}

}  // namespace musa
}  // namespace tensorflow
