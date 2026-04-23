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

#include "mu/device/peer_access.h"

#include <musa_runtime.h>

#include <mutex>
#include <unordered_map>

namespace tensorflow {
namespace musa {
namespace {

// Mutex + cache table. File-local to keep the header's include list
// limited to <cstdint>, <string>, <utility>, <vector>. The table is
// small (O(N^2) worst case, typically just a handful of pairs), so a
// single lock is cheap enough.
std::mutex g_lock;
std::unordered_map<std::uint64_t, PeerAccessEntry> g_cache;

}  // namespace

PeerAccess& PeerAccess::Instance() {
  // Leaked singleton — matches AllocatorConfig / DriverApi. Avoids
  // static-destruction-order issues when Python / TF tear down in a
  // surprising sequence.
  static PeerAccess* inst = new PeerAccess();
  return *inst;
}

PeerAccess::PeerAccess() {
  int n = 0;
  musaError_t e = musaGetDeviceCount(&n);
  if (e != musaSuccess) {
    // Swallow the last error so TF-side code doesn't see a stale
    // `musaGetLastError()` from this probe on the next call path.
    (void)musaGetLastError();
    device_count_ = 0;
    return;
  }
  device_count_ = n;
}

bool PeerAccess::CanAccessPeer(int from, int to) {
  if (device_count_ == 0) return false;
  if (from < 0 || from >= device_count_ || to < 0 || to >= device_count_) {
    return false;
  }
  if (from == to) return true;

  const std::uint64_t key = Key(from, to);
  {
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_cache.find(key);
    if (it != g_cache.end() && it->second.can_access != -1) {
      return it->second.can_access == 1;
    }
  }

  int flag = 0;
  musaError_t e = musaDeviceCanAccessPeer(&flag, from, to);
  if (e != musaSuccess) {
    // Don't cache failures: they are rare and usually transient
    // (e.g. driver not yet initialized) so caching would turn a
    // one-off hiccup into a permanent "no".
    (void)musaGetLastError();
    return false;
  }

  std::lock_guard<std::mutex> lk(g_lock);
  auto& entry = g_cache[key];
  entry.from = from;
  entry.to = to;
  entry.can_access = flag ? 1 : 0;
  return flag != 0;
}

bool PeerAccess::EnablePeerAccess(int from, int to) {
  if (!CanAccessPeer(from, to)) {
    // CanAccessPeer handles range/equality edge cases; when it
    // returns false either the pair is unsupported or out of range.
    // Either way, nothing to enable.
    return from == to;
  }

  // Check cache: already-enabled pairs avoid touching the driver.
  {
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_cache.find(Key(from, to));
    if (it != g_cache.end() && it->second.enabled) return true;
  }

  // `musaDeviceEnablePeerAccess` acts on the *current* device, so
  // we must switch to `from` first. Save/restore the caller's
  // current device so this probe is side-effect free.
  int saved = 0;
  if (musaGetDevice(&saved) != musaSuccess) {
    (void)musaGetLastError();
    saved = -1;
  }

  musaError_t e = musaSetDevice(from);
  if (e != musaSuccess) {
    (void)musaGetLastError();
    if (saved >= 0) (void)musaSetDevice(saved);
    return false;
  }

  e = musaDeviceEnablePeerAccess(to, /*flags=*/0);
  // musaErrorPeerAccessAlreadyEnabled is success from our perspective.
  // We compare against the numeric value rather than the symbolic
  // name because some older MUSA runtime headers don't export the
  // constant; value 704 matches CUDA's and MUSA's convention.
  constexpr musaError_t kAlreadyEnabled = static_cast<musaError_t>(704);
  bool ok = (e == musaSuccess) || (e == kAlreadyEnabled);
  if (!ok) (void)musaGetLastError();

  if (saved >= 0) (void)musaSetDevice(saved);

  if (ok) {
    std::lock_guard<std::mutex> lk(g_lock);
    auto& entry = g_cache[Key(from, to)];
    entry.from = from;
    entry.to = to;
    entry.can_access = 1;
    entry.enabled = true;
  }
  return ok;
}

std::vector<PeerAccessEntry> PeerAccess::Snapshot() const {
  std::vector<PeerAccessEntry> out;
  std::lock_guard<std::mutex> lk(g_lock);
  out.reserve(g_cache.size());
  for (const auto& kv : g_cache) out.push_back(kv.second);
  return out;
}

}  // namespace musa
}  // namespace tensorflow
