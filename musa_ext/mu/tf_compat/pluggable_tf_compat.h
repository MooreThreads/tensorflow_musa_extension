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

#ifndef TENSORFLOW_MUSA_MU_TF_COMPAT_PLUGGABLE_TF_COMPAT_H_
#define TENSORFLOW_MUSA_MU_TF_COMPAT_PLUGGABLE_TF_COMPAT_H_

#include <musa_runtime.h>

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {
class DeviceBase;

namespace musa {
namespace tf_compat {

// Maps TensorFlow's StreamExecutor `Stream` to the underlying MUSA stream
// handle. Prefers `GpuStreamMemberHack()` when present; otherwise uses
// `GpuStreamHack()` and may unwrap StreamExecutor-plugin `SP_Stream_st`
// (magic in `mu/musa_plugin_sp_stream.h`). Must stay isolated here so kernel
// code can depend on a single stable header.
musaStream_t GpuStreamFromTfStream(stream_executor::Stream* stream);

// Reads `gpu_id` from `DeviceBase::tensorflow_gpu_device_info()` when
// present; otherwise returns -1.
int GpuIdFromDeviceBase(const DeviceBase* base);

}  // namespace tf_compat
}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_TF_COMPAT_PLUGGABLE_TF_COMPAT_H_
