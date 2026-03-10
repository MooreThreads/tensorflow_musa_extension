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

#ifndef TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_

#include <musa_runtime.h>

#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

// MUSA-specific implementation of the StreamInterface.
class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}

  ~MusaStream() override {
    if (musa_stream_ != nullptr) {
      musaStreamDestroy(musa_stream_);
    }
  }

  // Returns the underlying musaStream_t handle.
  musaStream_t handle() const { return musa_stream_; }

  // Sets the stream handle (used for ownership transfer scenarios).
  void set_handle(musaStream_t stream) { musa_stream_ = stream; }

  // Returns the GPU stream associated with this platform's stream
  // implementation. This is used by higher-level abstractions that have
  // explicit dependencies on CUDA/MUSA runtime types.
  void* GpuStreamHack() override { return static_cast<void*>(musa_stream_); }

  // Returns the address of the internal stream handle.
  // See the above comment on GpuStreamHack.
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&musa_stream_);
  }

 private:
  musaStream_t musa_stream_;

  SE_DISALLOW_COPY_AND_ASSIGN(MusaStream);
};

// Helper function to extract the MUSA stream from a Stream.
inline MusaStream* AsMusaStream(Stream* stream) {
  return static_cast<MusaStream*>(stream->implementation());
}

// Helper function to extract the musaStream_t handle from a Stream.
inline musaStream_t AsMusaStreamValue(Stream* stream) {
  auto* musa_stream = AsMusaStream(stream);
  return musa_stream ? musa_stream->handle() : nullptr;
}

}  // namespace musa
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
