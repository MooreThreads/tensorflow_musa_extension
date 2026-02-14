#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_EVENT_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_EVENT_H_

#include <musa_runtime.h>

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaEvent : public internal::EventInterface {
 public:
  MusaEvent() : event_(nullptr), init_(false) {}

  ~MusaEvent() override {
    if (init_ && event_) {
      musaEventDestroy(event_);
    }
  }

  bool Init() {
    musaError_t err = musaEventCreateWithFlags(&event_, musaEventDisableTiming);
    init_ = (err == musaSuccess);
    return init_;
  }

  Event::Status PollForStatus() {
    musaError_t err = musaEventQuery(event_);

    if (err == musaSuccess) return Event::Status::kComplete;
    if (err == musaErrorNotReady) return Event::Status::kPending;
    return Event::Status::kError;
  }

  musaEvent_t handle() { return event_; }

 private:
  musaEvent_t event_;
  bool init_;
};

}  // namespace musa
}  // namespace stream_executor
#endif
