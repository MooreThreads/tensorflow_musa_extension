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

#ifndef TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_
#define TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_

#include <musa_runtime.h>

#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

// MUSA-specific implementation of the TimerInterface.
class MusaTimer : public internal::TimerInterface {
 public:
  MusaTimer() : start_event_(nullptr), stop_event_(nullptr), elapsed_nanoseconds_(0) {}

  ~MusaTimer() override {
    if (start_event_ != nullptr) {
      musaEventDestroy(start_event_);
    }
    if (stop_event_ != nullptr) {
      musaEventDestroy(stop_event_);
    }
  }

  // Initializes the timer by creating the start and stop events.
  // Returns true on success.
  bool Init() {
    if (musaEventCreate(&start_event_) != musaSuccess) {
      return false;
    }
    if (musaEventCreate(&stop_event_) != musaSuccess) {
      musaEventDestroy(start_event_);
      start_event_ = nullptr;
      return false;
    }
    return true;
  }

  // Starts the timer on the given stream.
  bool Start(musaStream_t stream) {
    if (start_event_ == nullptr) {
      if (!Init()) {
        return false;
      }
    }
    return musaEventRecord(start_event_, stream) == musaSuccess;
  }

  // Stops the timer on the given stream.
  bool Stop(musaStream_t stream) {
    if (stop_event_ == nullptr) {
      return false;
    }
    if (musaEventRecord(stop_event_, stream) != musaSuccess) {
      return false;
    }
    // Synchronize to get accurate timing
    if (musaEventSynchronize(stop_event_) != musaSuccess) {
      return false;
    }
    // Calculate elapsed time
    float milliseconds = 0.0f;
    if (musaEventElapsedTime(&milliseconds, start_event_, stop_event_) != musaSuccess) {
      return false;
    }
    elapsed_nanoseconds_ = static_cast<uint64>(milliseconds * 1e6f);
    return true;
  }

  // Returns the number of microseconds elapsed in a completed timer.
  uint64 Microseconds() const override {
    return elapsed_nanoseconds_ / 1000;
  }

  // Returns the number of nanoseconds elapsed in a completed timer.
  uint64 Nanoseconds() const override {
    return elapsed_nanoseconds_;
  }

 private:
  musaEvent_t start_event_;
  musaEvent_t stop_event_;
  uint64 elapsed_nanoseconds_;

  SE_DISALLOW_COPY_AND_ASSIGN(MusaTimer);
};

}  // namespace musa
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_
