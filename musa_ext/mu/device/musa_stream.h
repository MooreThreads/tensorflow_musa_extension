#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_

#include <cstring>
#include <musa_runtime.h>

#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor {
namespace musa {

class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}
  ~MusaStream() override {}
  musaStream_t GetStream() const { return musa_stream_; }

  tsl::Status BlockHostUntilDone_DEBUG(Stream* stream) {
    musaError_t result = musaStreamSynchronize(musa_stream_);
    if (result != musaSuccess) {
      // Minimal compatibility fix: teardown may hit already-destroyed stream.
      if (result == musaErrorContextIsDestroyed) {
        return tsl::OkStatus();
      }
      const char* err_msg = musaGetErrorString(result);
      if (err_msg != nullptr &&
          std::strstr(err_msg, "already destroyed") != nullptr) {
        return tsl::OkStatus();
      }
      return tsl::errors::Internal("Sync Failed");
    }
    return tsl::OkStatus();
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