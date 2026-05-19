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

#include "mu/collective/musa_collective_runtime.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

bool MusaCollectiveCommunicatorCache::Lookup(const std::string& cache_key,
                                              void** out_opaque) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(cache_key);
  if (it == entries_.end()) {
    return false;
  }
  if (out_opaque != nullptr) {
    *out_opaque = it->second;
  }
  return true;
}

bool MusaCollectiveCommunicatorCache::Insert(const std::string& cache_key,
                                            void* opaque) {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.emplace(cache_key, opaque).second;
}

void MusaCollectiveCommunicatorCache::Erase(const std::string& cache_key) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.erase(cache_key);
}

void MusaCollectiveCommunicatorCache::AbortAll(int error_code) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
  (void)error_code;
}

std::size_t MusaCollectiveCommunicatorCache::SizeForTest() const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

Status MusaCollectiveMcclPlaceholder::SmokeAllReduceFloat(
    const MusaCollectiveGroupSpec&, musaStream_t, float*, int count) {
  // Wired to MCCL once ABI + TF collective adapter land.
  return errors::Unimplemented(
      "MUSA MCCL collective smoke not wired (placeholder). count=", count);
}

}  // namespace musa
}  // namespace tensorflow
