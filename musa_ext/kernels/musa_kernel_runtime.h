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

#ifndef TENSORFLOW_MUSA_KERNELS_MUSA_KERNEL_RUNTIME_H_
#define TENSORFLOW_MUSA_KERNELS_MUSA_KERNEL_RUNTIME_H_

#include <musa_runtime.h>

#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace musa {

// Unifies the historical C++ `MusaDevice` path with the PluggableDevice path:
// - `musa_device` is non-null on the default C++ `DeviceFactory` + `MusaDevice`
//   registration path.
// - On the SE/Pluggable path, `musa_device` is null. Resolve `device_id`
//   from `(legacy) GpuDeviceInfo` where available; resolve `stream` solely from
//   `DeviceContext`'s StreamExecutor (`GpuStreamHack()`). Do **not** guess a
//   stream per ordinal — multiple concurrent streams per device cannot be
//   disambiguated without coupling to TF internals.
struct MusaKernelRuntimeView {
  MusaDevice* musa_device = nullptr;
  int device_id = -1;
  musaStream_t stream = nullptr;
  // Valid only when `musa_device != nullptr` (C++ MUSA path).
  ::musa::dnn::Handle* mudnn_handle = nullptr;
};

// Resolves the custom C++ `MusaDevice` registered by this plugin's historical
// path (`DeviceFactory::Register` + `tf.load_op_library`).
//
// TensorFlow's PluggableDevice path uses a different `Device` implementation;
// for those devices this returns nullptr. Kernels that need `MusaDevice` for
// muDNN must use `MUSA_OP_REQUIRES_CPP_MUSA_DEVICE` or a runtime view with a
// non-null `mudnn_handle` (future SE wiring).
inline MusaDevice* TryGetMusaDeviceFromContext(OpKernelContext* context) {
  if (context == nullptr) return nullptr;
  DeviceBase* base = context->device();
  if (base == nullptr) return nullptr;
  return dynamic_cast<MusaDevice*>(base);
}

MusaKernelRuntimeView QueryMusaKernelRuntimeView(OpKernelContext* context);
musaStream_t GetMusaStreamForKernelContext(OpKernelContext* context);
int GetMusaDeviceIdForKernelContext(OpKernelContext* context);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_MUSA_KERNEL_RUNTIME_H_
