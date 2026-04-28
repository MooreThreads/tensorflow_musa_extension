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

#include "kernels/musa_kernel_runtime.h"

#include <musa_runtime.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace tensorflow {
namespace musa {

namespace {
musaStream_t MusaStreamFromStreamExecutorPluggable(stream_executor::Stream* s) {
  if (s == nullptr) return nullptr;
  stream_executor::internal::StreamInterface* impl = s->implementation();
  if (impl == nullptr) return nullptr;
  return static_cast<musaStream_t>(impl->GpuStreamHack());
}
}  // namespace

MusaKernelRuntimeView QueryMusaKernelRuntimeView(OpKernelContext* context) {
  MusaKernelRuntimeView v;
  if (context == nullptr) {
    return v;
  }
  const DeviceBase* base = context->device();
  int device_from_info = -1;
  if (base != nullptr) {
    auto* ginfo = base->tensorflow_gpu_device_info();
    if (ginfo != nullptr) {
      device_from_info = ginfo->gpu_id;
    }
  }
  v.musa_device = TryGetMusaDeviceFromContext(context);
  if (v.musa_device != nullptr) {
    v.device_id = v.musa_device->get_device_id();
    v.stream = v.musa_device->GetStream();
    v.mudnn_handle = &v.musa_device->mudnn_handle();
    return v;
  }
  v.device_id = device_from_info;
  DeviceContext* dctx = context->op_device_context();
  if (dctx != nullptr) {
    v.stream = MusaStreamFromStreamExecutorPluggable(dctx->stream());
  }
  return v;
}

musaStream_t GetMusaStreamForKernelContext(OpKernelContext* context) {
  return QueryMusaKernelRuntimeView(context).stream;
}

int GetMusaDeviceIdForKernelContext(OpKernelContext* context) {
  return QueryMusaKernelRuntimeView(context).device_id;
}

}  // namespace musa
}  // namespace tensorflow
