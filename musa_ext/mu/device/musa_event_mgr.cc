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

#include "musa_event_mgr.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

MusaEventMgr::MusaEventMgr(int device_id, int32 polling_active_delay_usecs,
                           int32 num_threads)
    : device_id_(device_id),
      polling_active_delay_usecs_(polling_active_delay_usecs),
      stop_polling_(false),
      threadpool_(Env::Default(), "musa_event_mgr",
                  static_cast<int32>(num_threads)) {
  StartPollingLoop();
}

MusaEventMgr::~MusaEventMgr() {
  // Set shutdown flag first to prevent new callbacks from being scheduled
  shutting_down_.store(true, std::memory_order_release);

  StopPollingLoop();

  // At this point, PollLoop has exited. All events that were in used_events_
  // have either been:
  // 1. Processed by PollLoop's final FreeMemory (during shutdown path)
  // 2. Left in used_events_ if PollLoop didn't process them
  //
  // We need to clean up any remaining events without querying them,
  // as the underlying streams/resources may have been destroyed.

  ToFreeVector to_free;
  {
    mutex_lock l(mu_);
    // Move all remaining events to to_free without querying
    for (auto& iu : used_events_) {
      to_free.push_back(std::move(iu));
    }
    used_events_.clear();
  }

  // Execute callbacks and destroy events synchronously
  for (const auto& iu : to_free) {
    if (iu.event != nullptr) {
      DestroyEvent(iu.event);
    }
    if (iu.func != nullptr) {
      musaSetDevice(device_id_);
      iu.func();
    } else if (iu.has_status_func) {
      musaSetDevice(device_id_);
      iu.status_func(::tsl::OkStatus());
    }
  }

  // Clear free_events_ - these are events that were recycled and
  // are safe to destroy
  std::vector<musaEvent_t> events_to_destroy;
  {
    mutex_lock l(mu_);
    events_to_destroy.swap(free_events_);
  }
  for (musaEvent_t event : events_to_destroy) {
    DestroyEvent(event);
  }
}

musaEvent_t MusaEventMgr::CreateEvent() {
  musaEvent_t event;
  musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
  if (err != musaSuccess) {
    LOG(ERROR) << "Failed to create MUSA event: " << musaGetErrorString(err);
    return nullptr;
  }
  return event;
}

void MusaEventMgr::DestroyEvent(musaEvent_t event) {
  if (event != nullptr) {
    musaError_t err = musaEventDestroy(event);
    // Only log as WARNING for errors other than invalid resource handle
    // which can happen during shutdown when events are cleaned up
    if (err != musaSuccess && err != musaErrorInvalidResourceHandle) {
      LOG(WARNING) << "Failed to destroy MUSA event: "
                   << musaGetErrorString(err);
    }
  }
}

void MusaEventMgr::ThenExecute(musaStream_t stream,
                               std::function<void()> func) {
  ToFreeVector to_free;
  {
    mutex_lock l(mu_);
    QueueFunc(stream, std::move(func));
    PollEvents(false, &to_free);
  }
  FreeMemory(to_free);
}

void MusaEventMgr::ThenExecuteWithStatus(musaStream_t stream,
                                         std::function<void(Status)> func) {
  ToFreeVector to_free;
  {
    mutex_lock l(mu_);
    QueueInUse(stream, {nullptr, nullptr, std::move(func), true});
    PollEvents(false, &to_free);
  }
  FreeMemory(to_free);
}

void MusaEventMgr::QueueInUse(musaStream_t stream, InUse in_use) {
  musaEvent_t event;
  if (!free_events_.empty()) {
    event = free_events_.back();
    free_events_.pop_back();
  } else {
    event = CreateEvent();
    if (event == nullptr) {
      if (in_use.func != nullptr) {
        in_use.func();
      } else if (in_use.has_status_func) {
        in_use.status_func(errors::Internal("Failed to create MUSA event"));
      }
      return;
    }
  }

  musaError_t err = musaEventRecord(event, stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "Failed to record MUSA event: " << musaGetErrorString(err);
    free_events_.push_back(event);
    if (in_use.func != nullptr) {
      in_use.func();
    } else if (in_use.has_status_func) {
      in_use.status_func(errors::Internal("Failed to record MUSA event"));
    }
    return;
  }

  in_use.event = event;
  used_events_.push_back(std::move(in_use));
  events_pending_.notify_all();
}

void MusaEventMgr::PollEvents(bool is_dedicated_poller, ToFreeVector* to_free) {
  // Out-of-order completion: iterate through list and process ready events
  // This avoids Head-of-Line blocking where a single slow event delays all
  // subsequent callbacks
  auto it = used_events_.begin();
  while (it != used_events_.end()) {
    InUse& iu = *it;
    musaEvent_t event = iu.event;

    musaError_t err = musaEventQuery(event);
    if (err == musaErrorNotReady) {
      ++it;
      continue;
    } else if (err != musaSuccess) {
      LOG(WARNING) << "MUSA event query failed: " << musaGetErrorString(err);
    }

    to_free->push_back(std::move(iu));
    it = used_events_.erase(it);
  }
}

void MusaEventMgr::FreeMemory(const ToFreeVector& to_free) {
  // First, collect all events to be recycled under lock
  // This prevents race conditions with other threads accessing free_events_
  {
    mutex_lock l(mu_);
    for (const auto& iu : to_free) {
      if (iu.event != nullptr) {
        free_events_.push_back(iu.event);
      }
    }
  }

  // Execute callbacks outside the lock to avoid blocking
  for (const auto& iu : to_free) {
    if (iu.func != nullptr) {
      // Inject device context before executing callback in threadpool thread
      // Background threads lose device context, causing silent failures
      auto user_func = iu.func;
      int device_id = device_id_;

      // During shutdown, execute callbacks synchronously to ensure
      // all callbacks complete before destruction
      if (shutting_down_.load(std::memory_order_acquire)) {
        musaSetDevice(device_id);
        user_func();
      } else {
        threadpool_.Schedule([user_func, device_id]() {
          musaSetDevice(device_id);
          user_func();
        });
      }
    } else if (iu.has_status_func) {
      Status status = ::tsl::OkStatus();
      auto status_func = iu.status_func;
      int device_id = device_id_;

      // During shutdown, execute callbacks synchronously
      if (shutting_down_.load(std::memory_order_acquire)) {
        musaSetDevice(device_id);
        status_func(status);
      } else {
        threadpool_.Schedule([status, status_func, device_id]() {
          musaSetDevice(device_id);
          status_func(status);
        });
      }
    }
  }
}

void MusaEventMgr::PollLoop() {
  ToFreeVector to_free;
  while (true) {
    {
      mutex_lock l(mu_);
      while (!stop_polling_ && used_events_.empty()) {
        events_pending_.wait(l);
      }

      if (stop_polling_) {
        // During shutdown, don't query events - just move them all to to_free
        // This avoids "invalid resource handle" errors when the underlying
        // resources may have already been destroyed
        if (shutting_down_.load(std::memory_order_acquire)) {
          for (auto& iu : used_events_) {
            to_free.push_back(std::move(iu));
          }
          used_events_.clear();
        } else {
          PollEvents(true, &to_free);
        }
        break;
      }

      PollEvents(true, &to_free);
    }

    FreeMemory(to_free);
    to_free.clear();

    if (polling_active_delay_usecs_ > 0) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }

  // Final FreeMemory call - during shutdown, destroy events directly
  // instead of adding to free_events_ to avoid double-free
  if (shutting_down_.load(std::memory_order_acquire)) {
    for (const auto& iu : to_free) {
      if (iu.event != nullptr) {
        DestroyEvent(iu.event);
      }
      // Execute callbacks synchronously during shutdown
      if (iu.func != nullptr) {
        musaSetDevice(device_id_);
        iu.func();
      } else if (iu.has_status_func) {
        musaSetDevice(device_id_);
        iu.status_func(::tsl::OkStatus());
      }
    }
  } else {
    FreeMemory(to_free);
  }

  if (polling_stopped_) {
    polling_stopped_->Notify();
  }
}

void MusaEventMgr::StartPollingLoop() {
  polling_stopped_.reset(new Notification());
  stop_polling_ = false;

  // Use dedicated std::thread for polling loop instead of threadpool
  // This prevents thread starvation when callbacks are slow
  polling_thread_ = std::thread(&MusaEventMgr::PollLoop, this);
}

void MusaEventMgr::StopPollingLoop() {
  {
    mutex_lock l(mu_);
    stop_polling_ = true;
    events_pending_.notify_all();
  }
  if (polling_stopped_ != nullptr) {
    polling_stopped_->WaitForNotification();
  }

  if (polling_thread_.joinable()) {
    polling_thread_.join();
  }
}

}  // namespace musa
}  // namespace tensorflow
