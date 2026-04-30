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

#ifndef TENSORFLOW_MUSA_MU_COLLECTIVE_MUSA_COLLECTIVE_RUNTIME_H_
#define TENSORFLOW_MUSA_MU_COLLECTIVE_MUSA_COLLECTIVE_RUNTIME_H_

#include <musa_runtime.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace musa {

// Planned integration target: CollectiveOps → StreamExecutor → MCCL.
// TensorFlow Core collective APIs (`CollectiveExecutor`, `NcclManager`, graph
// collectives) differ across TF minors; kernels must not include those headers.
// Operations enqueue MCCL only on the current collective/compute stream (never
// guess streams by ordinal).

struct MusaCollectiveGroupSpec {
  std::string group_key;
  std::string instance_key;
  int global_rank = -1;
  int local_rank = -1;
  int world_size = -1;
};

//
// Lightweight communicator cache keyed by opaque group/instance markers.
// Backend handles are intentionally generic until MCCL exposes stable C ABI.
//
class MusaCollectiveCommunicatorCache {
 public:
  using AbortFn = std::function<void(int error_code)>;

  bool Lookup(const std::string& cache_key, void** out_opaque);
  bool Insert(const std::string& cache_key, void* opaque);
  void Erase(const std::string& cache_key);
  void AbortAll(int error_code);
  std::size_t SizeForTest() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, void*> entries_;
};

class MusaCollectiveMcclPlaceholder {
 public:
  Status SmokeAllReduceFloat(const MusaCollectiveGroupSpec&, musaStream_t,
                             float*, int count);
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_COLLECTIVE_MUSA_COLLECTIVE_RUNTIME_H_
