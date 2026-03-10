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

#ifndef TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_

#include <musa_runtime.h>

#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

// MUSA-specific implementation of the EventInterface.
class MusaEvent : public internal::EventInterface {
 public:
  MusaEvent() : event_(nullptr), initialized_(false) {}

  ~MusaEvent() override { Destroy(); }

  // Initializes the event with timing disabled (for synchronization only).
  bool Init() {
    if (initialized_) {
      return true;
    }
    musaError_t err = musaEventCreateWithFlags(&event_, musaEventDisableTiming);
    initialized_ = (err == musaSuccess);
    return initialized_;
  }

  // Destroys the event.
  void Destroy() {
    if (initialized_ && event_ != nullptr) {
      musaEventDestroy(event_);
      event_ = nullptr;
      initialized_ = false;
    }
  }

  // Returns true if the event has been successfully initialized.
  bool IsInitialized() const { return initialized_; }

  // Returns the underlying musaEvent_t handle.
  musaEvent_t handle() const { return event_; }

  // Polls the event status.
  port::Status PollForStatus();

 private:
  musaEvent_t event_;
  bool initialized_;

  SE_DISALLOW_COPY_AND_ASSIGN(MusaEvent);
};

}  // namespace musa
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
