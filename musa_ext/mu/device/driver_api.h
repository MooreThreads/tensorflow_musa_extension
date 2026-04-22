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

// Dynamic loader for MUSA driver entry points that are not surfaced by
// the runtime API. Needed by the Virtual Memory Management (VMM)
// primitives the expandable-segments allocator relies on (plan §5.3):
//
//   muMemAddressReserve / muMemAddressFree
//   muMemCreate / muMemRelease
//   muMemMap / muMemUnmap
//   muMemSetAccess
//   muMemGetAllocationGranularity
//   muDeviceGetAttribute  (for the VMM-support capability probe)
//   muGetErrorString       (for diagnostics)
//
// We intentionally avoid a build-time dependency on `<musa.h>`: a
// subset of clusters runs an older driver whose `libmusa.so` does not
// export the VMM symbols. Going through `dlopen`/`dlsym` lets the core
// library keep working on those hosts — the only consequence of a
// missing symbol is that `expandable_segments:true` is silently
// downgraded to the classic `musaMalloc` path. This mirrors
// `torch_musa`'s approach in `csrc/core/driver_api.{h,cpp}`.
//
// Threading
// ---------
// Initialization is idempotent and guarded by `std::call_once`, so
// concurrent callers see a fully populated table after the first call
// returns. All function pointers are `nullptr` on hosts where
// `libmusa.so` is missing or lacks the symbol, and callers must
// explicitly check `IsVmmAvailable()` before dereferencing them.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_DRIVER_API_H_
#define TENSORFLOW_MUSA_MU_DEVICE_DRIVER_API_H_

#include <cstddef>
#include <cstdint>

namespace tensorflow {
namespace musa {

// Minimal redeclaration of the MUSA driver types the VMM APIs need.
// Keeping them here instead of including `<musa.h>` lets consumers of
// this header (the allocator, the _C Python extension) stay free of a
// direct driver-header dependency. The typedefs MUST match the ABI
// published by `libmusa.so`.
using MUdeviceptr_t = std::uint64_t;
using MUmemGenericAllocationHandle_t = unsigned long long;

// Location: keep enum values stable with <musa.h>.
struct MuMemLocation {
  int type;  // 0x1 = MU_MEM_LOCATION_TYPE_DEVICE.
  int id;
};

// Allocation properties struct. Layout mirrors `MUmemAllocationProp_st`
// from `<musa.h>` so a zero-initialized instance of this struct can be
// handed directly to `muMemCreate`. Do NOT reorder fields or change
// their sizes without cross-checking the driver ABI — a mismatch here
// would silently corrupt driver state.
struct MuMemAllocationProp {
  int type;                    // MU_MEM_ALLOCATION_TYPE_PINNED = 0x1
  int requested_handle_types;  // 0 for no export.
  MuMemLocation location;
  void* win32_handle_metadata;  // Unused on Linux, kept for ABI.
  struct {
    unsigned char compression_type;
    unsigned char gpu_direct_rdma_capable;
    unsigned short usage;
    unsigned char reserved[4];
  } alloc_flags;
};

// Access descriptor mirroring `MUmemAccessDesc_st`.
struct MuMemAccessDesc {
  MuMemLocation location;
  int flags;  // MU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3.
};

// Constants in line with `<musa.h>` enums, duplicated so callers do
// not need the driver header.
constexpr int kMuMemLocationTypeDevice = 0x1;
constexpr int kMuMemAllocationTypePinned = 0x1;
constexpr int kMuMemAccessFlagsReadWrite = 0x3;
constexpr int kMuMemAllocGranularityMinimum = 0x0;
// MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102.
constexpr int kMuDeviceAttrVmmSupported = 102;

// Table of driver function pointers. Any field may be `nullptr` when
// the host's `libmusa.so` does not export it; never dereference without
// first calling `IsVmmAvailable()` (or checking each pointer yourself).
struct DriverApi {
  // Address space reservation.
  int (*MemAddressReserve)(MUdeviceptr_t* ptr, std::size_t size,
                           std::size_t alignment, MUdeviceptr_t addr,
                           unsigned long long flags) = nullptr;
  int (*MemAddressFree)(MUdeviceptr_t ptr, std::size_t size) = nullptr;

  // Physical memory handle lifecycle.
  int (*MemCreate)(MUmemGenericAllocationHandle_t* handle, std::size_t size,
                   const MuMemAllocationProp* prop,
                   unsigned long long flags) = nullptr;
  int (*MemRelease)(MUmemGenericAllocationHandle_t handle) = nullptr;

  // Mapping handle → virtual range.
  int (*MemMap)(MUdeviceptr_t ptr, std::size_t size, std::size_t offset,
                MUmemGenericAllocationHandle_t handle,
                unsigned long long flags) = nullptr;
  int (*MemUnmap)(MUdeviceptr_t ptr, std::size_t size) = nullptr;

  // Accessibility.
  int (*MemSetAccess)(MUdeviceptr_t ptr, std::size_t size,
                      const MuMemAccessDesc* desc, std::size_t count) = nullptr;

  // Granularity query.
  int (*MemGetAllocationGranularity)(std::size_t* granularity,
                                     const MuMemAllocationProp* prop,
                                     int option) = nullptr;

  // Capability probe + diagnostics.
  int (*DeviceGetAttribute)(int* pi, int attrib, int dev) = nullptr;
  int (*GetErrorString)(int error, const char** pStr) = nullptr;
};

// Returns the shared driver API table, initializing it on first call.
// The table is a process-wide singleton owned by libmusa_core.so.
const DriverApi& GetDriverApi();

// True iff libmusa.so was opened successfully AND every VMM symbol
// needed by `ExpandableSegment` is non-null. Cheap to call repeatedly
// (returns a cached bool after the first call).
bool IsVmmAvailable();

// True iff the driver reports that the given device supports VMM (via
// MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED). Cached per
// device to keep the probe off the hot path. Returns false when
// `IsVmmAvailable()` is false.
bool IsVmmSupportedForDevice(int ordinal);

// Formats a driver error code as a human-readable string. Prefers
// `muGetErrorString` when available; otherwise returns
// "unknown driver error <code>". Never allocates; the returned
// string's storage is caller-owned by having a bounded static buffer
// semantics — callers must copy if retention is needed beyond the
// immediate scope.
const char* DriverErrorString(int code);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_DRIVER_API_H_
