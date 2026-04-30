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
#include <string>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

namespace {

std::mutex g_mu;
std::unordered_map<int32_t, MusaSeDeviceRuntimeState> g_devices;

bool Live(const MusaSeDeviceRuntimeState& s) { return s.ref_count > 0; }

void MaybeFillPciStub(int32_t ordinal, MusaSeDeviceRuntimeState* st) {
  if (st == nullptr || !st->pci_bus_id.empty()) return;
  st->pci_bus_id = std::string("ordinal:") + std::to_string(ordinal);
}

}  // namespace

void MusaSeRegistryOnDeviceCreated(int32_t ordinal) {
  std::lock_guard<std::mutex> lock(g_mu);
  MusaSeDeviceRuntimeState& st = g_devices[ordinal];
  st.ordinal = ordinal;
  ++st.ref_count;
  MaybeFillPciStub(ordinal, &st);
}

void MusaSeRegistryOnDeviceDestroyed(int32_t ordinal) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_devices.find(ordinal);
  if (it == g_devices.end()) {
    return;
  }
  if (--(it->second.ref_count) <= 0) {
    it->second.mudnn_by_stream.clear();
    it->second.collective_runtime_opaque.reset();
    g_devices.erase(it);
  }
}

::musa::dnn::Handle* MusaSeRegistryEnsureMudnnForDevice(int32_t ordinal,
                                                        musaStream_t stream) {
  if (ordinal < 0 || stream == nullptr) {
    return nullptr;
  }

  const uintptr_t stream_key = reinterpret_cast<uintptr_t>(stream);

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_devices.find(ordinal);
  if (it == g_devices.end() || !Live(it->second)) {
    return nullptr;
  }

  MusaSeDeviceRuntimeState& st = it->second;
  MusaSePerStreamMudnnSlot& slot = st.mudnn_by_stream[stream_key];

  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err != musaSuccess) {
    LOG(ERROR) << "MusaSeRegistryEnsureMudnnForDevice: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
    return nullptr;
  }

  if (!slot.handle && !slot.init_failed) {
    slot.handle.reset(new ::musa::dnn::Handle());
    ::musa::dnn::Status s = slot.handle->SetStream(stream);
    if (s != ::musa::dnn::Status::SUCCESS) {
      slot.handle.reset();
      slot.init_failed = true;
      LOG(ERROR) << "MuDNN Handle init failed for ordinal " << ordinal;
      return nullptr;
    }
  }

  if (slot.init_failed || !slot.handle) {
    return nullptr;
  }

  ::musa::dnn::Status ss = slot.handle->SetStream(stream);
  if (ss != ::musa::dnn::Status::SUCCESS) {
    LOG(ERROR) << "MuDNN SetStream failed for ordinal " << ordinal;
    return nullptr;
  }

  return slot.handle.get();
}

bool MusaSeRegistryHasLiveDeviceForTest(int32_t ordinal) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_devices.find(ordinal);
  if (it == g_devices.end()) return false;
  return Live(it->second);
}

std::size_t MusaSeRegistrySizeForTest() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_devices.size();
}

void MusaSeRegistryResetForTest() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_devices.clear();
}

}  // namespace musa
}  // namespace tensorflow
