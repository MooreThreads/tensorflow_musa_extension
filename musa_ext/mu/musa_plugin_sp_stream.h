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

#ifndef TENSORFLOW_MUSA_MU_MUSA_PLUGIN_SP_STREAM_H_
#define TENSORFLOW_MUSA_MU_MUSA_PLUGIN_SP_STREAM_H_

#include <cstdint>

#include <musa_runtime.h>

// Mark StreamExecutor-plugin stream wrappers (`SP_Stream`). `GpuStreamHack` may
// return either `(void*)(musaStream_t)` (C++ `MusaStream`) **or** a heap
// `SP_Stream_st*`; identify the latter with this 64-bit tag (avoid driver API probes).
constexpr uint64_t kMusaPluginSpStreamMagic =
    UINT64_C(0x4D55534153455053);  // ASCII "MUASEPS"

struct SP_Stream_st {
  uint64_t magic = 0;
  musaStream_t stream{};
};

#endif  // TENSORFLOW_MUSA_MU_MUSA_PLUGIN_SP_STREAM_H_
