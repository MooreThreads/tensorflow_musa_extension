#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_

#include <musa_runtime.h>

#if __has_include("tensorflow/stream_executor/platform/port.h")
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#define MUSA_SE_STATUS port::Status
#define MUSA_SE_INTERNAL_STATUS(message) \
  port::Status(port::error::INTERNAL, message)
#define MUSA_SE_OK_STATUS() port::Status::OK()
#else
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_internal.h"
#define MUSA_SE_STATUS tsl::Status
#define MUSA_SE_INTERNAL_STATUS(message) tsl::errors::Internal(message)
#define MUSA_SE_OK_STATUS() tsl::OkStatus()
#endif

namespace stream_executor {
namespace musa {

class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}
  ~MusaStream() override {}
  musaStream_t GetStream() const { return musa_stream_; }

  MUSA_SE_STATUS BlockHostUntilDone_DEBUG(Stream* stream) {
    musaError_t result = musaStreamSynchronize(musa_stream_);
    if (result != musaSuccess) {
      return MUSA_SE_INTERNAL_STATUS("Sync Failed");
    }
    return MUSA_SE_OK_STATUS();
  }

  void* GpuStreamHack() override { return (void*)musa_stream_; }
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&musa_stream_);
  }

 private:
  musaStream_t musa_stream_;
};

}  // namespace musa
}  // namespace stream_executor

#endif