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

// Process-wide, per-device pool of `musaEvent_t` handles.
//
// Hot paths inside the plugin (PluggableDevice `create_event` / `destroy_event`
// callbacks, the host caching allocator, the staging pool) all need to create
// short-lived events to synchronize recycling of resources. `musaEventCreate`
// and `musaEventDestroy` each take a driver round-trip that shows up
// measurably (0.3-0.5 ms / step on the prunedGraph inference workload) when
// EventMgr holds low-hundreds of in-flight events. This pool recycles the
// handles to eliminate those round-trips.
//
// All events go through `musaEventCreateWithFlags(musaEventDisableTiming)`,
// so record/wait are lightweight.
//
// Historically the pool lived as an anonymous-namespace class inside
// `musa_se_callbacks.cc`. It was promoted to a public header so other
// subsystems (caching allocators, telemetry) can share a single pool per
// device instead of each re-inventing its own.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_EVENT_POOL_H_
#define TENSORFLOW_MUSA_MU_DEVICE_EVENT_POOL_H_

#include <musa_runtime.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorflow {
namespace musa {

class EventPool {
 public:
  // Defined in event_pool.cc so that only one copy of the singleton storage
  // exists even when the plugin is split into multiple shared libraries
  // (libmusa_core.so, libmusa_plugin.so, tensorflow_musa._C.*.so). Every
  // dependent library must resolve `Instance` through core's export.
  static EventPool& Instance();

  // Returns a ready-to-record event for `ordinal`. On cache hit the pooled
  // handle is returned directly; on miss we fall back to the driver.
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
  // We intentionally do NOT call `musaEventQuery` before pooling: the
  // CUDA/MUSA programming model guarantees that `musaStreamWaitEvent`
  // captures the event's record state at the time of the wait call, so
  // later re-records on the same handle cannot affect earlier waits.
  // Skipping the query saves one driver round-trip per destroy, which is
  // the main reason the pool exists.
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

  // Opt-out via env var, useful for A/B testing correctness.
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
    std::lock_guard<std::mutex> lk(mu_);
    auto it = per_device_.find(ordinal);
    if (it != per_device_.end()) return *it->second;
    auto pd = std::unique_ptr<PerDevice>(new PerDevice);
    PerDevice* raw = pd.get();
    per_device_.emplace(ordinal, std::move(pd));
    return *raw;
  }

  std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<PerDevice>> per_device_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_EVENT_POOL_H_
