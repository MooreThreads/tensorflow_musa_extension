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

#ifndef TENSORFLOW_MUSA_MU_MUSA_RUNTIME_REGISTRY_H_
#define TENSORFLOW_MUSA_MU_MUSA_RUNTIME_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <musa_runtime.h>
#include <mudnn.h>

namespace tensorflow {
namespace musa {

struct MusaSePerStreamMudnnSlot {
  std::unique_ptr<::musa::dnn::Handle> handle;
  bool init_failed = false;
};

// Per-device PluggableDevice runtime: ref-counted from
// `plugin_create_device` / `plugin_destroy_device`. Does not map streams by
// ordinal alone (kernels must use `DeviceContext` / GpuStreamHack). muDNN
// handles are **per `musaStream_t` instance** on the SE path so concurrent ops
// on different streams do not share one mutable `Handle::SetStream` binding.

struct MusaSeDeviceRuntimeState {
  int32_t ordinal = -1;
  int ref_count = 0;

  std::unordered_map<uintptr_t, MusaSePerStreamMudnnSlot> mudnn_by_stream;

  std::string pci_bus_id;

  std::shared_ptr<void> collective_runtime_opaque;

  int last_collective_error_code = 0;
};

void MusaSeRegistryOnDeviceCreated(int32_t ordinal);
void MusaSeRegistryOnDeviceDestroyed(int32_t ordinal);

// Returns nullptr if the ordinal is not tracked (no live Pluggable device
// creation) or initialization failed. Binds `SetStream` on the **per-stream**
// handle for this `musaStream_t`.
::musa::dnn::Handle* MusaSeRegistryEnsureMudnnForDevice(int32_t ordinal,
                                                         musaStream_t stream);

bool MusaSeRegistryHasLiveDeviceForTest(int32_t ordinal);

std::size_t MusaSeRegistrySizeForTest();

// Test-only: clear all state (subprocess tests only).
void MusaSeRegistryResetForTest();

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_MUSA_RUNTIME_REGISTRY_H_
