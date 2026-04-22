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

// Expandable VMM-backed segment used when `TF_MUSA_ALLOC_CONF` has
// `expandable_segments:true` and the driver reports support.
//
// MVP semantics (commit C5)
// -------------------------
// The allocator's cache-miss path typically asks for ~2 MiB – a few
// GiB at a time. When expandable_segments is enabled each such
// request funnels through `ExpandableSegment::Create(size)`, which:
//
//   1. Rounds `size` up to the driver's minimum allocation granularity.
//   2. Reserves a private VA range of exactly that size via
//      `muMemAddressReserve` (one reservation per segment).
//   3. Allocates a pinned physical handle of the same size via
//      `muMemCreate`.
//   4. Maps the handle over the VA range via `muMemMap`.
//   5. Grants read/write access to the owning device via
//      `muMemSetAccess`.
//   6. Returns the VA as a `void*` plus the `ExpandableSegment` object
//      that owns the handle/reservation for later teardown.
//
// Release reverses the steps: `muMemUnmap` → `muMemRelease` →
// `muMemAddressFree`. All driver failures are reported as `false` from
// the create helper and logged to stderr with the driver error string.
//
// Why this MVP instead of a single giant pre-reservation
// ------------------------------------------------------
// The richer torch_musa model pre-reserves one VA range per device
// (~1.5× device memory) and grows the physical mapping in-place. That
// yields the best fragmentation wins but also rewrites the caching
// allocator's address-management. C5 keeps the caching allocator
// structurally unchanged — blocks live inside segments as before —
// and gets the plumbing in place (driver API, config, capability
// probe, Python surface, tests) on a one-segment-per-Create basis.
// A follow-up commit can upgrade `ExpandableSegment` to hold many
// handles over a single reservation without touching the allocator's
// outer loop.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_EXPANDABLE_SEGMENT_H_
#define TENSORFLOW_MUSA_MU_DEVICE_EXPANDABLE_SEGMENT_H_

#include <cstddef>
#include <cstdint>

#include "mu/device/driver_api.h"

namespace tensorflow {
namespace musa {

// Owns exactly one VA reservation, one physical handle, and the
// mapping between them. All methods are non-owning w.r.t. threads —
// the caller (the caching allocator) serializes across segments via
// its own mutex.
class ExpandableSegment {
 public:
  // Factory helper. Returns nullptr on any driver failure (including
  // OOM from muMemCreate or address-space exhaustion). `out_ptr` is
  // populated only on success; `out_actual_size` reflects the
  // granularity-rounded byte count (always >= `requested_bytes`).
  //
  // `device` is the MUSA device ordinal that will own the mapping and
  // get read-write access.
  //
  // Requires `IsVmmAvailable() && IsVmmSupportedForDevice(device)`;
  // otherwise returns nullptr without touching the driver.
  static ExpandableSegment* Create(int device, std::size_t requested_bytes,
                                   void** out_ptr,
                                   std::size_t* out_actual_size);

  // Reverse of Create: unmap, release handle, free VA reservation.
  // Safe to call on a partially-initialized segment (e.g. after a
  // mid-Create failure); no-ops skip what isn't there.
  ~ExpandableSegment();

  ExpandableSegment(const ExpandableSegment&) = delete;
  ExpandableSegment& operator=(const ExpandableSegment&) = delete;

  int device() const { return device_; }
  void* ptr() const { return reinterpret_cast<void*>(va_); }
  std::size_t size() const { return size_; }
  std::size_t granularity() const { return granularity_; }

 private:
  ExpandableSegment() = default;

  int device_ = -1;
  MUdeviceptr_t va_ = 0;
  std::size_t size_ = 0;
  std::size_t granularity_ = 0;
  MUmemGenericAllocationHandle_t handle_ = 0;

  // Each step's success tracks what needs to be undone in the dtor.
  bool va_reserved_ = false;
  bool handle_created_ = false;
  bool mapped_ = false;
};

// Round `size` up to a multiple of `granularity`. Exposed for tests
// that want to verify address alignment behavior.
std::size_t RoundUpToGranularity(std::size_t size, std::size_t granularity);

// Returns the driver-reported minimum allocation granularity for
// pinned device memory on `device`. Useful both inside this module
// and as a Python-side sanity check. Returns 0 on failure.
std::size_t QueryMinAllocationGranularity(int device);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_EXPANDABLE_SEGMENT_H_
