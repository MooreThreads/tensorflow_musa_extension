/* Copyright 2025 The TensorFlow MUSA Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_EVENT_MGR_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_EVENT_MGR_H_

#include <musa_runtime.h>

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace musa {

// Manages MUSA events for asynchronous callback notification.
//
// The callback provided to MusaEventMgr::ThenExecute must not block or take a
// long time. If it does, performance may be impacted and device memory may be
// exhausted.
class MusaEventMgr {
 public:
  // Constructor takes the device_id and optional configuration.
  // Default num_threads is 8 to prevent thread starvation when handling
  // long-running callbacks (e.g., large memcpy operations).
  MusaEventMgr(int device_id, int32 polling_active_delay_usecs = 100,
               int32 num_threads = 8);
  virtual ~MusaEventMgr();

  // Execute func when all pending stream actions have completed.
  // func must be brief and non-blocking since it executes in the threadpool.
  void ThenExecute(musaStream_t stream, std::function<void()> func);

  // Alternative API: Queue a callback that will be called when the event
  // associated with the stream completes.
  void ThenExecuteWithStatus(musaStream_t stream,
                             std::function<void(Status)> func);

 private:
  struct InUse {
    musaEvent_t event;
    std::function<void()> func;
    std::function<void(Status)> status_func;
    bool has_status_func;
  };

  typedef gtl::InlinedVector<InUse, 4> ToFreeVector;

  const int device_id_;
  const int32 polling_active_delay_usecs_;
  mutex mu_;
  condition_variable events_pending_ TF_GUARDED_BY(mu_);

  musaEvent_t CreateEvent();
  void DestroyEvent(musaEvent_t event);

  void QueueInUse(musaStream_t stream, InUse in_use)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueFunc(musaStream_t stream, std::function<void()> func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, std::move(func), nullptr, false});
  }

  void PollEvents(bool is_dedicated_poller, ToFreeVector* to_free)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void PollLoop();
  void StartPollingLoop();
  void StopPollingLoop();

  void FreeMemory(const ToFreeVector& to_free);

  std::vector<musaEvent_t> free_events_ TF_GUARDED_BY(mu_);

  // Use std::list instead of std::deque for out-of-order completion.
  // This avoids Head-of-Line blocking where a slow event delays all
  // subsequent callbacks.
  std::list<InUse> used_events_ TF_GUARDED_BY(mu_);

  bool stop_polling_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;

  // Dedicated thread for polling loop to prevent threadpool starvation
  std::thread polling_thread_;

  thread::ThreadPool threadpool_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_EVENT_MGR_H_
