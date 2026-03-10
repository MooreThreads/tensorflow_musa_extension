/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "mu/device/musa_event.h"

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {
namespace musa {

port::Status MusaEvent::PollForStatus() {
  if (!initialized_ || event_ == nullptr) {
    return port::Status(port::error::INTERNAL, "Event not initialized");
  }

  musaError_t err = musaEventQuery(event_);
  if (err == musaSuccess) {
    return port::Status::OK();
  } else if (err == musaErrorNotReady) {
    return port::Status(port::error::UNAVAILABLE, "Event not ready");
  } else {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("musaEventQuery failed: ",
                                     musaGetErrorString(err)));
  }
}

}  // namespace musa
}  // namespace stream_executor
