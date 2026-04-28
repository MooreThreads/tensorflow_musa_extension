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

#include "mu/tf_compat/pluggable_tf_compat.h"

#include "mu/musa_plugin_sp_stream.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace tensorflow {
namespace musa {
namespace tf_compat {

namespace {

// `GpuStreamHack()` may return `(void*)musa_stream_t` (bare handle bits) for
// native `MusaStream` — those must not be dereferenced as `SP_Stream_st`.
// Always prefer `GpuStreamMemberHack()` when overridden (MusaStream / GpuStream):
// it yields the handle without ambiguous pointer interpretation.
musaStream_t ResolveStreamFromHack(void* hack) {
  if (hack == nullptr) return nullptr;
  const SP_Stream_st* wrapper = reinterpret_cast<const SP_Stream_st*>(hack);
  if (wrapper->magic == kMusaPluginSpStreamMagic) {
    return wrapper->stream;
  }
  return reinterpret_cast<musaStream_t>(hack);
}

}  // namespace

musaStream_t GpuStreamFromTfStream(stream_executor::Stream* stream) {
  if (stream == nullptr) return nullptr;
  stream_executor::internal::StreamInterface* impl = stream->implementation();
  if (impl == nullptr) return nullptr;
  void** member_hack = impl->GpuStreamMemberHack();
  if (member_hack != nullptr) {
    return *reinterpret_cast<musaStream_t*>(member_hack);
  }
  return ResolveStreamFromHack(impl->GpuStreamHack());
}

int GpuIdFromDeviceBase(const DeviceBase* base) {
  if (base == nullptr) return -1;
  auto* ginfo = base->tensorflow_gpu_device_info();
  if (ginfo == nullptr) return -1;
  return ginfo->gpu_id;
}

}  // namespace tf_compat
}  // namespace musa
}  // namespace tensorflow
