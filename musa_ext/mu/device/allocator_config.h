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

// Parser for the `TF_MUSA_ALLOC_CONF` env variable.
//
// The config format matches torch_musa / PyTorch's
// `PYTORCH_CUDA_ALLOC_CONF` so ops teams can translate runbooks 1:1:
//
//   TF_MUSA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
//
// Keys are case-insensitive; values are parsed according to each key's
// expected type. Unknown keys are logged once to stderr and ignored —
// that keeps forward compatibility easy (adding a new toggle never
// breaks older consumers that set it).
//
// Only `expandable_segments` is honored in commit C5. The other
// fields are parsed and stored for future extensions (max-split,
// roundup-power2-divisions, garbage_collection_threshold) so the
// surface stays stable when those are implemented.

#ifndef TENSORFLOW_MUSA_MU_DEVICE_ALLOCATOR_CONFIG_H_
#define TENSORFLOW_MUSA_MU_DEVICE_ALLOCATOR_CONFIG_H_

#include <cstdint>
#include <string>

namespace tensorflow {
namespace musa {

class AllocatorConfig {
 public:
  // Process-wide singleton. The first call parses
  // `TF_MUSA_ALLOC_CONF`; subsequent calls are free.
  static const AllocatorConfig& Instance();

  // True iff `expandable_segments:true|True|1|yes` is present in the
  // env var. Effective only when the driver supports VMM
  // (IsVmmAvailable + IsVmmSupportedForDevice); consumers should
  // confirm support before acting on this flag.
  bool expandable_segments() const { return expandable_segments_; }

  // Upper bound (in bytes) above which the allocator will not split a
  // cached block for reuse. 0 = unlimited (the default). Not yet
  // enforced in C5; reserved for a later commit.
  std::uint64_t max_split_size_bytes() const { return max_split_size_bytes_; }

  // Power-of-two sub-bucket factor for size-class rounding. 0 = off
  // (the default). Reserved for a later commit.
  int roundup_power2_divisions() const { return roundup_power2_divisions_; }

  // GC threshold as a fraction of reserved_bytes. 0.0 disables.
  // Reserved for a later commit; parsed here so env surface is stable.
  double garbage_collection_threshold() const {
    return garbage_collection_threshold_;
  }

  // Raw config string, useful for diagnostics.
  const std::string& raw() const { return raw_; }

 private:
  AllocatorConfig() = default;
  void ParseFromEnv();

  bool expandable_segments_ = false;
  std::uint64_t max_split_size_bytes_ = 0;
  int roundup_power2_divisions_ = 0;
  double garbage_collection_threshold_ = 0.0;
  std::string raw_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_ALLOCATOR_CONFIG_H_
