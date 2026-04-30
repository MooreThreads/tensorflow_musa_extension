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

#include "mu/musa_runtime_registry.h"
#include "mu/tf_compat/pluggable_tf_compat.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace musa {

MusaKernelRuntimeView QueryMusaKernelRuntimeView(OpKernelContext* context) {
  MusaKernelRuntimeView v;
  if (context == nullptr) {
    return v;
  }
  DeviceBase* base = context->device();
  const int device_from_info = tf_compat::GpuIdFromDeviceBase(base);

  if (base != nullptr) {
    v.allocator = base->GetAllocator(AllocatorAttributes());
  }

  v.musa_device = TryGetMusaDeviceFromContext(context);
  if (v.musa_device != nullptr) {
    v.is_pluggable = false;
    v.device_id = v.musa_device->get_device_id();
    v.stream = v.musa_device->GetStream();
    v.mudnn_handle = &v.musa_device->mudnn_handle();
    return v;
  }

  v.is_pluggable = true;
  v.device_id = device_from_info;

  DeviceContext* dctx = context->op_device_context();
  if (dctx != nullptr) {
    v.stream = tf_compat::GpuStreamFromTfStream(dctx->stream());
  }

  if (v.device_id >= 0 && v.stream != nullptr) {
    ::musa::dnn::Handle* h =
        MusaSeRegistryEnsureMudnnForDevice(v.device_id, v.stream);
    v.mudnn_handle = h;
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
