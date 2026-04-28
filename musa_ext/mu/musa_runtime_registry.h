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

namespace tensorflow {
namespace musa {

// Tracks PluggableDevice SE `SP_Device` ordinal lifetimes from
// `plugin_create_device` / `plugin_destroy_device` (per-device runtime state
// hook; no per-stream map — multiple streams per ordinal cannot be resolved
// safely for kernel dispatch without `DeviceContext`).

void MusaSeRegistryOnDeviceCreated(int32_t ordinal);
void MusaSeRegistryOnDeviceDestroyed(int32_t ordinal);
std::size_t MusaSeRegistrySizeForTest();

// Test-only: clear all state (subprocess tests only).
void MusaSeRegistryResetForTest();

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_MUSA_RUNTIME_REGISTRY_H_
