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
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/public/version.h"
#if __has_include("tensorflow/stream_executor/stream.h")
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#else
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_internal.h"
#endif

namespace tensorflow {
namespace musa {
namespace tf_compat {

namespace {

musaStream_t ResolvePluginCStream(
    stream_executor::internal::StreamInterface* impl) {
  auto* cstream = dynamic_cast<stream_executor::CStream*>(impl);
  if (cstream == nullptr) return nullptr;
  SP_Stream wrapper = cstream->Handle();
  if (wrapper == nullptr || wrapper->magic != kMusaPluginSpStreamMagic) {
    return nullptr;
  }
  return wrapper->stream;
}

}  // namespace

musaStream_t GpuStreamFromTfStream(stream_executor::Stream* stream) {
  if (stream == nullptr) return nullptr;
  stream_executor::internal::StreamInterface* impl = stream->implementation();
  if (impl == nullptr) return nullptr;
  musaStream_t plugin_stream = ResolvePluginCStream(impl);
  if (plugin_stream != nullptr) return plugin_stream;
  void** member_hack = impl->GpuStreamMemberHack();
  if (member_hack == nullptr || *member_hack == nullptr) return nullptr;
  return *reinterpret_cast<musaStream_t*>(member_hack);
}

int GpuIdFromDeviceBase(const DeviceBase* base) {
  if (base == nullptr) return -1;
#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 10)
  auto* ginfo = base->tensorflow_accelerator_device_info();
#else
  auto* ginfo = base->tensorflow_gpu_device_info();
#endif
  if (ginfo == nullptr) return -1;
  return ginfo->gpu_id;
}

}  // namespace tf_compat
}  // namespace musa
}  // namespace tensorflow
