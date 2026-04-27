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

#include "mu/device/musa_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace musa {

// Resolves the custom C++ `MusaDevice` registered by this plugin's historical
// path (`DeviceFactory::Register` + `tf.load_op_library`).
//
// TensorFlow's PluggableDevice path uses a different `Device` implementation;
// for those devices this returns nullptr. Kernels that need `MusaDevice`
// must fail or no-op until SE-only execution is fully wired.
inline MusaDevice* TryGetMusaDeviceFromContext(OpKernelContext* context) {
  if (context == nullptr) return nullptr;
  DeviceBase* base = context->device();
  if (base == nullptr) return nullptr;
  return dynamic_cast<MusaDevice*>(base);
}

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_MUSA_KERNEL_RUNTIME_H_
