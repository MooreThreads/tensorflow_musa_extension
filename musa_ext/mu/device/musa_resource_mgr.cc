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

#include "mu/device/musa_resource_mgr.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

MusaResourceMgr& MusaResourceMgr::Instance() {
  static MusaResourceMgr instance;
  return instance;
}

MusaResourceMgr::~MusaResourceMgr() {
  // Best-effort cleanup at process exit. Handles may have already been
  // released via explicit Shutdown() calls, but the unordered_map destructor
  // takes care of the rest without touching the driver (which may already be
  // shutting down).
  for (auto& kv : per_device_) {
    if (kv.second && kv.second->mublas) {
      mublasDestroy(kv.second->mublas);
      kv.second->mublas = nullptr;
    }
    if (kv.second) kv.second->mudnn.reset();
  }
  per_device_.clear();
}

MusaResourceMgr::PerDevice* MusaResourceMgr::LookupOrCreate(int device_id) {
  // Fast path (reader): no write to the map, just a lookup under the lock.
  // Entries are never relocated after emplace(), so the returned raw pointer
  // is stable for the process lifetime.
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = per_device_.find(device_id);
    if (it != per_device_.end()) return it->second.get();
  }

  // Slow path: build the handles outside the lock so the (potentially
  // expensive) mudnn / mublas constructors never block other devices.
  auto entry = std::unique_ptr<PerDevice>(new PerDevice());
  musaSetDevice(device_id);
  entry->mudnn.reset(new ::musa::dnn::Handle());
  mublasStatus_t s = mublasCreate(&entry->mublas);
  if (s != MUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "[MUSA] mublasCreate failed for device " << device_id
               << " status=" << static_cast<int>(s);
    entry->mublas = nullptr;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto it = per_device_.find(device_id);
  if (it != per_device_.end()) {
    // Lost the race; drop our duplicate.
    if (entry->mublas) mublasDestroy(entry->mublas);
    return it->second.get();
  }
  PerDevice* raw = entry.get();
  per_device_.emplace(device_id, std::move(entry));
  return raw;
}

void MusaResourceMgr::Init(int device_id) { (void)LookupOrCreate(device_id); }

void MusaResourceMgr::Shutdown(int device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = per_device_.find(device_id);
  if (it == per_device_.end()) return;

  musaSetDevice(device_id);
  if (it->second && it->second->mublas) {
    mublasDestroy(it->second->mublas);
    it->second->mublas = nullptr;
  }
  if (it->second) it->second->mudnn.reset();
  per_device_.erase(it);
}

// Per-thread cache of the last (device, mudnn/mublas) binding.
//
// A single TF session typically dispatches a few-thousand kernel launches
// per step onto a small, fixed set of executor threads. Caching the
// resolved PerDevice* plus the most-recently-bound stream turns the hot
// path through GetMudnnHandle / GetMublasHandle into (in the common case)
// two branches and no atomic ops at all. The shared state we rely on --
// PerDevice* -- is stable for the process lifetime; see
// MusaResourceMgr::LookupOrCreate.
namespace {
// Sentinel distinct from musaStreamDefault / any legal stream handle,
// so the first call on a new thread always re-binds.
inline musaStream_t SentinelStream() {
  return reinterpret_cast<musaStream_t>(static_cast<intptr_t>(-1));
}
}  // namespace

::musa::dnn::Handle& MusaResourceMgr::GetMudnnHandle(int device_id,
                                                     musaStream_t stream) {
  static thread_local int tl_device_id = -1;
  static thread_local PerDevice* tl_pd = nullptr;
  static thread_local musaStream_t tl_last_stream = SentinelStream();

  if (tl_device_id != device_id || tl_pd == nullptr) {
    tl_pd = LookupOrCreate(device_id);
    tl_device_id = device_id;
    tl_last_stream = SentinelStream();
  }
  // Rebinding mudnn's stream touches internal library state; skip it when
  // the caller's stream is unchanged (true on every iteration of a TF1
  // inference loop after the first).
  if (tl_pd->mudnn && stream != tl_last_stream) {
    tl_pd->mudnn->SetStream(stream);
    tl_last_stream = stream;
  }
  return *tl_pd->mudnn;
}

mublasHandle_t MusaResourceMgr::GetMublasHandle(int device_id,
                                                musaStream_t stream) {
  // Separate TL cache from the mudnn path so the two handles can carry
  // different streams without thrashing each other's "last stream" memo.
  static thread_local int tl_device_id = -1;
  static thread_local PerDevice* tl_pd = nullptr;
  static thread_local musaStream_t tl_last_stream = SentinelStream();

  if (tl_device_id != device_id || tl_pd == nullptr) {
    tl_pd = LookupOrCreate(device_id);
    tl_device_id = device_id;
    tl_last_stream = SentinelStream();
  }
  if (tl_pd->mublas && stream != tl_last_stream) {
    mublasSetStream(tl_pd->mublas, stream);
    tl_last_stream = stream;
  }
  return tl_pd->mublas;
}

}  // namespace musa
}  // namespace tensorflow
