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

// Peer-to-peer access cache (plan §3.8 / §4.3 C3 — "optional" tier).
//
// On multi-device hosts, two devices can often DMA into each other's
// memory directly via `musaDeviceEnablePeerAccess`, which lets later
// D2D traffic skip the staging-through-host detour. The MUSA runtime
// exposes two primitives for this:
//
//   * `musaDeviceCanAccessPeer(&flag, from, to)` — capability probe
//   * `musaDeviceEnablePeerAccess(peer, 0)` — side-effect: the
//     *calling* device gains access to `peer`'s memory. The call is
//     asymmetric: enabling 0→1 does not enable 1→0.
//
// Both calls are idempotent but hit the driver; neither is free.
// `PeerAccess` caches both results so repeated queries (e.g. every
// cross-device tensor) stay cheap, and records which ordered pairs
// have actually been enabled so we can expose a clear snapshot.
//
// Scope (what this file does NOT do):
//   * It does *not* change any PluggableDevice callback. The active
//     `memcpy_dtod` still assumes src/dst live on the same device,
//     and adding peer-aware dispatch there is a separate, bigger
//     commit (it needs allocator-side ordinal lookup for each
//     `SP_DeviceMemoryBase`).
//   * It does *not* extend expandable-segment VMM mappings with
//     peer `muMemSetAccess` calls; that work is gated on the above
//     and on demand from multi-GPU users.
//
// What it *does* give the rest of the stack is a single, thread-safe
// place to answer "can 2 talk to 5?" and "has 2→5 been enabled?".
// That's enough to unblock user-facing Python helpers
// (`can_access_peer`, `enable_peer_access`) that are useful even
// without the deeper wiring.

#ifndef MUSA_EXT_MU_DEVICE_PEER_ACCESS_H_
#define MUSA_EXT_MU_DEVICE_PEER_ACCESS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tensorflow {
namespace musa {

// A single (from, to) ordered pair entry in the cache. `can_access`
// carries a tri-state: unknown (-1), false (0), true (1); we keep
// the tri-state so first-time queries and persistent "no, cannot"
// answers stay distinguishable in the snapshot.
struct PeerAccessEntry {
  int from;
  int to;
  int can_access;  // -1 unknown, 0 false, 1 true
  bool enabled;    // musaDeviceEnablePeerAccess(from -> to) succeeded
};

// Process-wide singleton (leaked on teardown, matching the rest of
// the core; see allocator_config.cc for the rationale).
class PeerAccess {
 public:
  static PeerAccess& Instance();

  // Number of devices reported by the MUSA runtime at init time. Zero
  // on hosts where the runtime fails to initialize (e.g. no driver);
  // all query methods short-circuit with `false` in that case.
  int device_count() const noexcept { return device_count_; }

  // Returns true iff `from` can DMA into `to`'s memory. Result is
  // cached after the first query per pair. Degenerate self-access
  // (from == to) always returns true — the runtime accepts it too.
  // Ordinals outside [0, device_count()) return false and are *not*
  // cached so later resizes stay cheap.
  bool CanAccessPeer(int from, int to);

  // Enable `from -> to` peer access if the capability is present
  // and not already enabled. Returns true if access is live after
  // the call (already-enabled pairs count as success). Errors (e.g.
  // `musaErrorPeerAccessAlreadyEnabled`) are squashed to true so
  // repeated calls stay idempotent.
  bool EnablePeerAccess(int from, int to);

  // Snapshot the known entries (includes only pairs we've touched;
  // untouched pairs are absent rather than marked "unknown" to keep
  // the output small on large clusters).
  std::vector<PeerAccessEntry> Snapshot() const;

 private:
  PeerAccess();
  PeerAccess(const PeerAccess&) = delete;
  PeerAccess& operator=(const PeerAccess&) = delete;

  // Packs two int ordinals into a single 64-bit key for the cache
  // map. Stays cheap to compute and lookup; collisions are impossible
  // in the int range.
  static constexpr std::uint64_t Key(int from, int to) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from))
            << 32) |
           static_cast<std::uint32_t>(to);
  }

  int device_count_ = 0;
  // Intentionally no mutex here; all public methods take a lock
  // defined in the .cc file. Keeping the lock private reduces the
  // header's include surface.
};

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_EXT_MU_DEVICE_PEER_ACCESS_H_
