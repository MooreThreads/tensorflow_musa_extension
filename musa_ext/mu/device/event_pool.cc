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

// Out-of-line definition of `EventPool::Instance()`. Keeping the storage in
// a single translation unit (compiled only into libmusa_core.so) is what
// guarantees a unique pool per process across the plugin / pybind / future
// helpers. All other shared libraries must resolve their `Instance()` call
// through core's exported symbol rather than instantiating a fresh static
// local of their own.

#include "mu/device/event_pool.h"

namespace tensorflow {
namespace musa {

EventPool& EventPool::Instance() {
  // Intentionally leaked: static-local destruction at process exit runs
  // after `libmusart.so` may have been unloaded in the split-library
  // layout, which turned `musaEventDestroy` in ~PerDevice into a crash.
  // A leak at process exit is free; correctness is not.
  static EventPool* kInst = new EventPool();
  return *kInst;
}

}  // namespace musa
}  // namespace tensorflow
