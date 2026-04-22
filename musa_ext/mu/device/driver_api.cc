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

#include "mu/device/driver_api.h"

#include <dlfcn.h>
#include <musa_runtime.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tensorflow {
namespace musa {

namespace {

// Known install locations for libmusa.so. We try each in order; the
// first dlopen that succeeds wins. On dev-container images the
// Debian-style `/usr/lib/x86_64-linux-gnu/libmusa.so` is typical; on
// vendor installs we also find `${MUSA_HOME}/lib/libmusa.so`.
constexpr std::array<const char*, 5> kCandidatePaths = {
    "libmusa.so",
    "libmusa.so.1",
    "/usr/lib/x86_64-linux-gnu/libmusa.so",
    "/usr/local/musa/lib/libmusa.so",
    "/usr/local/musa/lib64/libmusa.so",
};

// Resolve a symbol from the driver handle, swallowing missing-symbol
// errors. Missing symbols are a supported outcome — on older drivers
// some VMM APIs are absent, and we handle that by keeping the
// corresponding pointer null and surfacing it through IsVmmAvailable().
template <typename Fn>
Fn ResolveSymbol(void* handle, const char* name) {
  if (handle == nullptr) return nullptr;
  // Clear any stale dlerror state first so the post-call check is sound.
  (void)dlerror();
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (sym == nullptr || err != nullptr) return nullptr;
  return reinterpret_cast<Fn>(sym);
}

struct LoadState {
  void* handle = nullptr;
  DriverApi api{};
  bool vmm_available = false;
  std::string load_error;

  std::mutex cap_mu;
  std::unordered_map<int, bool> vmm_supported_by_device;
};

LoadState& State() {
  // Leaked on purpose — the driver handle must outlive static
  // destructors because other globals (allocator, Python singletons)
  // may still reach into it during process teardown.
  static LoadState* s = new LoadState();
  return *s;
}

void InitDriverApi() {
  auto& st = State();
  void* h = nullptr;
  for (const char* path : kCandidatePaths) {
    // RTLD_NOW so missing dependencies surface immediately rather than
    // only when a specific VMM call is exercised.
    h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h != nullptr) break;
  }
  if (h == nullptr) {
    const char* err = dlerror();
    st.load_error =
        err != nullptr ? std::string(err) : std::string("dlopen failed");
    return;
  }
  st.handle = h;

  // Resolve the full VMM surface. VMM availability requires ALL of
  // them — partial resolution indicates a driver that is too old for
  // expandable segments.
  DriverApi& api = st.api;
  api.MemAddressReserve =
      ResolveSymbol<decltype(api.MemAddressReserve)>(h, "muMemAddressReserve");
  api.MemAddressFree =
      ResolveSymbol<decltype(api.MemAddressFree)>(h, "muMemAddressFree");
  api.MemCreate = ResolveSymbol<decltype(api.MemCreate)>(h, "muMemCreate");
  api.MemRelease = ResolveSymbol<decltype(api.MemRelease)>(h, "muMemRelease");
  api.MemMap = ResolveSymbol<decltype(api.MemMap)>(h, "muMemMap");
  api.MemUnmap = ResolveSymbol<decltype(api.MemUnmap)>(h, "muMemUnmap");
  api.MemSetAccess =
      ResolveSymbol<decltype(api.MemSetAccess)>(h, "muMemSetAccess");
  api.MemGetAllocationGranularity =
      ResolveSymbol<decltype(api.MemGetAllocationGranularity)>(
          h, "muMemGetAllocationGranularity");

  // Capability + diagnostics are independently useful; fail-open even
  // if they are missing, but without them we can't probe per-device
  // support, so VMM stays disabled.
  api.DeviceGetAttribute = ResolveSymbol<decltype(api.DeviceGetAttribute)>(
      h, "muDeviceGetAttribute");
  api.GetErrorString =
      ResolveSymbol<decltype(api.GetErrorString)>(h, "muGetErrorString");

  const bool have_vmm = api.MemAddressReserve != nullptr &&
                        api.MemAddressFree != nullptr &&
                        api.MemCreate != nullptr && api.MemRelease != nullptr &&
                        api.MemMap != nullptr && api.MemUnmap != nullptr &&
                        api.MemSetAccess != nullptr &&
                        api.MemGetAllocationGranularity != nullptr &&
                        api.DeviceGetAttribute != nullptr;
  st.vmm_available = have_vmm;
  if (!have_vmm) {
    st.load_error =
        "libmusa.so opened but one or more VMM symbols are missing; "
        "expandable_segments will remain disabled.";
  }
}

std::once_flag g_init_once;

}  // namespace

const DriverApi& GetDriverApi() {
  std::call_once(g_init_once, InitDriverApi);
  return State().api;
}

bool IsVmmAvailable() {
  std::call_once(g_init_once, InitDriverApi);
  return State().vmm_available;
}

bool IsVmmSupportedForDevice(int ordinal) {
  if (!IsVmmAvailable()) return false;
  auto& st = State();
  {
    std::lock_guard<std::mutex> lk(st.cap_mu);
    auto it = st.vmm_supported_by_device.find(ordinal);
    // Only cache *positive* results. The driver-level attribute query
    // returns MUSA_ERROR_NOT_INITIALIZED (rc=3) before the first
    // runtime call on this process, so a negative answer can simply
    // mean "we probed too early". Re-try on each call until we see
    // a stable True, then stop probing.
    if (it != st.vmm_supported_by_device.end() && it->second) return true;
  }
  // Force driver init through a harmless runtime call. musaSetDevice
  // is idempotent and cheap on the already-set device, and it
  // guarantees `muDeviceGetAttribute` won't fail with
  // MUSA_ERROR_NOT_INITIALIZED. We intentionally do NOT propagate a
  // failure here; the probe below will record the correct result if
  // init still doesn't work.
  (void)musaSetDevice(ordinal);
  (void)musaGetLastError();
  int value = 0;
  const int rc =
      st.api.DeviceGetAttribute(&value, kMuDeviceAttrVmmSupported, ordinal);
  const bool supported = (rc == 0) && (value != 0);
  std::lock_guard<std::mutex> lk(st.cap_mu);
  if (supported) st.vmm_supported_by_device[ordinal] = true;
  return supported;
}

const char* DriverErrorString(int code) {
  static thread_local char buf[64];
  auto& api = State().api;
  if (api.GetErrorString != nullptr) {
    const char* p = nullptr;
    if (api.GetErrorString(code, &p) == 0 && p != nullptr) return p;
  }
  std::snprintf(buf, sizeof(buf), "unknown driver error %d", code);
  return buf;
}

}  // namespace musa
}  // namespace tensorflow
