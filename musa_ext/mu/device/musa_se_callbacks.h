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

#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_SE_CALLBACKS_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_SE_CALLBACKS_H_

#include "mu/tf_compat.h"

namespace tensorflow {
namespace musa {

// Populates `se` with the full set of SP_StreamExecutor callbacks backed by
// the MUSA driver runtime. Mirrors the CUDA-equivalent surface in
// tensorflow/stream_executor/cuda/cuda_gpu_executor.cc as exposed via the
// PluggableDevice C API.
void PopulateStreamExecutor(SP_StreamExecutor* se);

// Populates `timer` with a no-op timer implementation. MUSA exposes timing
// through mudnn profilers; TensorFlow's SP_Timer is only used for stream
// timing in benchmarks, which we do not currently support.
void PopulateTimerFns(SP_TimerFns* timer);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_SE_CALLBACKS_H_
