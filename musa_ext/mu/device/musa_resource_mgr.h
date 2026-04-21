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

#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_RESOURCE_MGR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_RESOURCE_MGR_H_

#include <mublas.h>
#include <mudnn.h>
#include <musa_runtime.h>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace tensorflow {
namespace musa {

// Per-device MUSA resources (mudnn handle + mublas handle).
//
// Lifetime: MusaResourceMgr is a process-global singleton. Handles are created
// lazily per device ordinal on first use, and destroyed either via Shutdown()
// (driven by plugin_destroy_device in the PluggableDevice C API) or at process
// exit.
//
// Stream binding: mudnn's internal stream and mublas' internal stream are
// re-bound in GetMudnnHandle/GetMublasHandle whenever the caller's stream
// differs from the last stream we bound on this thread. This matches how
// TF's cuBLAS/cuDNN wrappers work under CUDA (tensorflow/stream_executor/
// cuda/cuda_blas.cc rebinds on every call) but skips the rebind when the
// stream is unchanged -- a common case for the TF1 single-compute-stream
// inference loop where this helper is called thousands of times per step.
//
// Hot-path design: after first-touch, GetMudnnHandle / GetMublasHandle are
// lock-free. The global mutex is only taken on the slow path that creates
// a new PerDevice entry for an unseen ordinal (once per device per process).
// A thread_local cache stores the resolved PerDevice* so repeated lookups
// from the same kernel-executor thread have no atomic ops at all.
class MusaResourceMgr {
 public:
  static MusaResourceMgr& Instance();

  // Lazily create mudnn + mublas handles for `device_id`. Safe to call from
  // multiple threads; repeated calls are no-ops.
  void Init(int device_id);

  // Release handles for `device_id`. Must be called while the device context
  // is still live. After Shutdown, a subsequent Init() will recreate handles.
  void Shutdown(int device_id);

  // Returns the per-device mudnn handle with its stream bound to `stream`.
  // The returned reference is stable for the lifetime of the device.
  ::musa::dnn::Handle& GetMudnnHandle(int device_id, musaStream_t stream);

  // Returns the per-device mublas handle with its stream bound to `stream`.
  mublasHandle_t GetMublasHandle(int device_id, musaStream_t stream);

 private:
  struct PerDevice {
    std::unique_ptr<::musa::dnn::Handle> mudnn;
    mublasHandle_t mublas = nullptr;
  };

  MusaResourceMgr() = default;
  ~MusaResourceMgr();
  MusaResourceMgr(const MusaResourceMgr&) = delete;
  MusaResourceMgr& operator=(const MusaResourceMgr&) = delete;

  // Locates / creates the PerDevice entry. On the first-touch slow path this
  // takes `mu_`; once the entry exists the returned pointer is stable for
  // the lifetime of the process (entries are never relocated, and Shutdown
  // is only triggered at plugin teardown). Callers therefore cache the
  // result in thread_local storage without re-locking.
  PerDevice* LookupOrCreate(int device_id);

  std::mutex mu_;
  std::unordered_map<int, std::unique_ptr<PerDevice>> per_device_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_RESOURCE_MGR_H_
