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

#include "mu/musa_runtime_registry.h"

#include <mutex>
#include <unordered_map>

namespace tensorflow {
namespace musa {
namespace {

std::mutex g_mu;
std::unordered_map<int32_t, int> g_by_ordinal;

}  // namespace

void MusaSeRegistryOnDeviceCreated(int32_t ordinal) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_by_ordinal[ordinal] += 1;
}

void MusaSeRegistryOnDeviceDestroyed(int32_t ordinal) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_by_ordinal.find(ordinal);
  if (it == g_by_ordinal.end()) {
    return;
  }
  if (--(it->second) <= 0) {
    g_by_ordinal.erase(it);
  }
}

std::size_t MusaSeRegistrySizeForTest() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_by_ordinal.size();
}

void MusaSeRegistryResetForTest() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_by_ordinal.clear();
}

}  // namespace musa
}  // namespace tensorflow
