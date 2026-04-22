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

// PluggableDevice entry point for the MUSA backend.
//
// This file implements the stable C API declared in
// tensorflow/c/experimental/stream_executor/stream_executor.h. TensorFlow
// (2.6.x through at least 2.16.x) invokes `SE_InitPlugin` when the plugin
// .so is loaded via `TF_LoadPluggableDeviceLibrary` (or auto-loaded from
// `$SITE_PACKAGES/tensorflow-plugins`). From there, TF calls our
// `plugin_*` callbacks to enumerate, create, and tear down MUSA devices.
// All TF header inclusion is centralized in mu/tf_compat.h for version
// compatibility management; see that file for version-gated features.
//
// The legacy `REGISTER_LOCAL_DEVICE_FACTORY` path (which required linking
// against internal core APIs and had ABI risk) is intentionally deleted.

#include "mu/device_register.h"

#include <musa_runtime.h>

#include <cstdint>
#include <cstring>

#include "mu/device/musa_resource_mgr.h"
#include "mu/device/musa_se_callbacks.h"
#include "mu/device/musa_telemetry.h"
// tf_compat.h already transitively included via mu/device_register.h and
// mu/device/musa_se_callbacks.h; re-include to keep intent explicit.
#include "mu/tf_compat.h"

namespace {

constexpr char kMusaPlatformName[] = "MUSA";
constexpr char kMusaDeviceType[] = "MUSA";

inline void SetOK(TF_Status* status) { TF_SetStatus(status, TF_OK, ""); }

}  // namespace

extern "C" {

void plugin_get_device_count(const SP_Platform* /*platform*/, int* count,
                             TF_Status* const status) {
  int n = 0;
  musaError_t err = musaGetDeviceCount(&n);
  if (err != musaSuccess) {
    // Don't fail plugin init if no devices are present — just report zero.
    *count = 0;
    SetOK(status);
    return;
  }
  *count = n;
  SetOK(status);
}

void plugin_create_device(const SP_Platform* /*platform*/,
                          SE_CreateDeviceParams* params, TF_Status* status) {
  SP_Device* device = params->device;
  device->struct_size = SP_DEVICE_STRUCT_SIZE;
  device->ordinal = params->ordinal;
  // Store the ordinal in device_handle too for cheap lookup from callbacks
  // that only get SP_Device*.
  device->device_handle =
      reinterpret_cast<void*>(static_cast<intptr_t>(params->ordinal));
  device->hardware_name = "MUSA Device";
  device->device_vendor = "Moore Threads";
  device->pci_bus_id = "";

  // Create per-device mudnn / mublas handles up front so that kernels can
  // fetch them without racing on first-use. Matches TF GPU's
  // GPUDeviceContext constructor which eagerly creates library handles.
  tensorflow::musa::MusaResourceMgr::Instance().Init(params->ordinal);

  SetOK(status);
}

void plugin_destroy_device(const SP_Platform* /*platform*/, SP_Device* device) {
  if (!device) return;
  tensorflow::musa::MusaResourceMgr::Instance().Shutdown(device->ordinal);
  device->device_handle = nullptr;
}

void plugin_create_device_fns(const SP_Platform* /*platform*/,
                              SE_CreateDeviceFnsParams* params,
                              TF_Status* status) {
  params->device_fns->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  params->device_fns->ext = nullptr;
  // NUMA/bandwidth/gflops are optional; leaving them unset forces TF to treat
  // them as "unknown", which matches TF GPU's behavior on non-NUMA hosts.
  params->device_fns->get_numa_node = nullptr;
  params->device_fns->get_memory_bandwidth = nullptr;
  params->device_fns->get_gflops = nullptr;
  SetOK(status);
}

void plugin_destroy_device_fns(const SP_Platform* /*platform*/,
                               SP_DeviceFns* /*device_fns*/) {}

void plugin_create_stream_executor(const SP_Platform* /*platform*/,
                                   SE_CreateStreamExecutorParams* params,
                                   TF_Status* status) {
  tensorflow::musa::PopulateStreamExecutor(params->stream_executor);
  SetOK(status);
}

void plugin_destroy_stream_executor(const SP_Platform* /*platform*/,
                                    SP_StreamExecutor* /*stream_executor*/) {}

void plugin_create_timer_fns(const SP_Platform* /*platform*/,
                             SP_TimerFns* timer, TF_Status* status) {
  tensorflow::musa::PopulateTimerFns(timer);
  SetOK(status);
}

void plugin_destroy_timer_fns(const SP_Platform* /*platform*/,
                              SP_TimerFns* /*timer_fns*/) {}

void plugin_destroy_platform(SP_Platform* /*platform*/) {}

void plugin_destroy_platform_fns(SP_PlatformFns* /*platform_fns*/) {}

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  params->struct_size = SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->major_version = SE_MAJOR;
  params->minor_version = SE_MINOR;
  params->patch_version = SE_PATCH;

  SP_Platform* platform = params->platform;
  std::memset(platform, 0, sizeof(*platform));
  platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  platform->name = kMusaPlatformName;
  platform->type = kMusaDeviceType;
  platform->supports_unified_memory = 0;
  // Delegate BFC pooling to TF core's PluggableDeviceBFCAllocator, matching
  // tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc behavior. Our
  // raw allocate/deallocate just wraps musaMalloc/musaFree, and TF grows the
  // pool on demand the same way it does for CUDA.
  platform->use_bfc_allocator = 1;

  SP_PlatformFns* pfns = params->platform_fns;
  std::memset(pfns, 0, sizeof(*pfns));
  pfns->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
  pfns->get_device_count = plugin_get_device_count;
  pfns->create_device = plugin_create_device;
  pfns->destroy_device = plugin_destroy_device;
  pfns->create_device_fns = plugin_create_device_fns;
  pfns->destroy_device_fns = plugin_destroy_device_fns;
  pfns->create_stream_executor = plugin_create_stream_executor;
  pfns->destroy_stream_executor = plugin_destroy_stream_executor;
  pfns->create_timer_fns = plugin_create_timer_fns;
  pfns->destroy_timer_fns = plugin_destroy_timer_fns;

  params->destroy_platform = plugin_destroy_platform;
  params->destroy_platform_fns = plugin_destroy_platform_fns;

  SetOK(status);
}

// Telemetry is process-global; retain the constructor/destructor pair so that
// loading the .so (whether via tf.load_pluggable_device_library or the
// tensorflow-plugins auto-load) still primes telemetry exactly once.
void __attribute__((constructor)) OnMusaPluginLoad() {
  auto config = ::tensorflow::musa::TelemetryConfig::FromEnv();
  if (config.enabled) {
    ::tensorflow::musa::MusaTelemetry::Instance().Initialize(config);
  }
}

void __attribute__((destructor)) OnMusaPluginUnload() {
  ::tensorflow::musa::MusaTelemetry::Instance().Shutdown();
}

}  // extern "C"

// Required by headers that still reference NAME_MTGPU; unused after refactor
// but kept for ABI compatibility with older Python loaders.
const char NAME_MTGPU[] = "MUSA";
