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

#include "mu/device/expandable_segment.h"

#include <cstdio>
#include <cstdlib>

namespace tensorflow {
namespace musa {

namespace {

// Fills `prop` with the canonical "pinned, device-local, no export"
// configuration. Returns by value to keep callers declarative.
MuMemAllocationProp MakeDeviceProp(int device) {
  MuMemAllocationProp prop{};
  prop.type = kMuMemAllocationTypePinned;
  prop.requested_handle_types = 0;  // No shareable handle export.
  prop.location.type = kMuMemLocationTypeDevice;
  prop.location.id = device;
  prop.win32_handle_metadata = nullptr;
  return prop;
}

bool VerboseEnabled() {
  static const bool k = []() {
    const char* e = std::getenv("TF_MUSA_ALLOC_VERBOSE_OOM");
    if (e == nullptr || *e == '\0') return false;
    return !(e[0] == '0' || e[0] == 'f' || e[0] == 'F');
  }();
  return k;
}

void LogDriverErr(const char* where, int rc) {
  if (!VerboseEnabled()) return;
  std::fprintf(stderr, "[MUSA VMM] %s failed: %s (rc=%d)\n", where,
               DriverErrorString(rc), rc);
  std::fflush(stderr);
}

}  // namespace

std::size_t RoundUpToGranularity(std::size_t size, std::size_t granularity) {
  if (granularity == 0) return size;
  return ((size + granularity - 1) / granularity) * granularity;
}

std::size_t QueryMinAllocationGranularity(int device) {
  const DriverApi& api = GetDriverApi();
  if (api.MemGetAllocationGranularity == nullptr) return 0;
  MuMemAllocationProp prop = MakeDeviceProp(device);
  std::size_t g = 0;
  int rc =
      api.MemGetAllocationGranularity(&g, &prop, kMuMemAllocGranularityMinimum);
  if (rc != 0) {
    LogDriverErr("muMemGetAllocationGranularity", rc);
    return 0;
  }
  return g;
}

ExpandableSegment* ExpandableSegment::Create(int device,
                                             std::size_t requested_bytes,
                                             void** out_ptr,
                                             std::size_t* out_actual_size) {
  if (out_ptr != nullptr) *out_ptr = nullptr;
  if (out_actual_size != nullptr) *out_actual_size = 0;

  if (!IsVmmAvailable() || !IsVmmSupportedForDevice(device)) return nullptr;
  if (requested_bytes == 0) return nullptr;

  const DriverApi& api = GetDriverApi();
  const std::size_t granularity = QueryMinAllocationGranularity(device);
  if (granularity == 0) return nullptr;
  const std::size_t size = RoundUpToGranularity(requested_bytes, granularity);

  auto* seg = new ExpandableSegment();
  seg->device_ = device;
  seg->size_ = size;
  seg->granularity_ = granularity;

  // (1) Reserve VA.
  int rc = api.MemAddressReserve(&seg->va_, size, granularity,
                                 /*addr=*/0, /*flags=*/0);
  if (rc != 0 || seg->va_ == 0) {
    LogDriverErr("muMemAddressReserve", rc);
    delete seg;  // nothing to undo.
    return nullptr;
  }
  seg->va_reserved_ = true;

  // (2) Create physical handle.
  MuMemAllocationProp prop = MakeDeviceProp(device);
  rc = api.MemCreate(&seg->handle_, size, &prop, /*flags=*/0);
  if (rc != 0) {
    LogDriverErr("muMemCreate", rc);
    delete seg;  // dtor releases the VA reservation.
    return nullptr;
  }
  seg->handle_created_ = true;

  // (3) Map handle → VA.
  rc = api.MemMap(seg->va_, size, /*offset=*/0, seg->handle_, /*flags=*/0);
  if (rc != 0) {
    LogDriverErr("muMemMap", rc);
    delete seg;  // dtor releases handle + VA.
    return nullptr;
  }
  seg->mapped_ = true;

  // (4) Enable R/W access from the owning device.
  MuMemAccessDesc desc{};
  desc.location.type = kMuMemLocationTypeDevice;
  desc.location.id = device;
  desc.flags = kMuMemAccessFlagsReadWrite;
  rc = api.MemSetAccess(seg->va_, size, &desc, 1);
  if (rc != 0) {
    LogDriverErr("muMemSetAccess", rc);
    delete seg;
    return nullptr;
  }

  if (out_ptr != nullptr) *out_ptr = reinterpret_cast<void*>(seg->va_);
  if (out_actual_size != nullptr) *out_actual_size = size;
  return seg;
}

ExpandableSegment::~ExpandableSegment() {
  const DriverApi& api = GetDriverApi();
  if (mapped_ && api.MemUnmap != nullptr) {
    int rc = api.MemUnmap(va_, size_);
    if (rc != 0) LogDriverErr("muMemUnmap (dtor)", rc);
  }
  if (handle_created_ && api.MemRelease != nullptr) {
    int rc = api.MemRelease(handle_);
    if (rc != 0) LogDriverErr("muMemRelease (dtor)", rc);
  }
  if (va_reserved_ && api.MemAddressFree != nullptr) {
    int rc = api.MemAddressFree(va_, size_);
    if (rc != 0) LogDriverErr("muMemAddressFree (dtor)", rc);
  }
}

}  // namespace musa
}  // namespace tensorflow
