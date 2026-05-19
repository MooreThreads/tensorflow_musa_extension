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

#include "pinned_memory_pool.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

static constexpr size_t kMinAllocationSize = 256;
static constexpr int32 kPollingIntervalUs = 100;

GPUPinnedMemoryPool::GPUPinnedMemoryPool(int device_id)
    : device_id_(device_id), stop_polling_(false) {
  musaSetDevice(device_id_);
  polling_thread_ = std::thread(&GPUPinnedMemoryPool::PollLoop, this);
}

GPUPinnedMemoryPool::~GPUPinnedMemoryPool() {
  {
    mutex_lock l(mu_);
    stop_polling_ = true;
    poll_cv_.notify_all();
  }

  if (polling_thread_.joinable()) {
    polling_thread_.join();
  }

  musaSetDevice(device_id_);

  // Process all pending frees under lock to ensure thread safety
  // and proper cleanup of events and memory
  mutex_lock l(mu_);

  // First, destroy all events in pending_frees_ and release memory directly
  // Do NOT move to free_list_ to avoid potential double-free issues
  for (auto& block : pending_frees_) {
    if (block.event != nullptr) {
      musaEventDestroy(block.event);
      block.event = nullptr;
    }
    // Release memory directly instead of moving to free_list_
    ReleaseBlock(block);
  }
  pending_frees_.clear();

  // Release all blocks in free_list_
  for (const auto& block : free_list_) {
    ReleaseBlock(block);
  }
  free_list_.clear();
}

void* GPUPinnedMemoryPool::Allocate(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  size_t alloc_size = bytes;
  if (alloc_size < kMinAllocationSize) {
    alloc_size = kMinAllocationSize;
  }

  // Try to find a reusable block from free_list_
  {
    mutex_lock l(mu_);

    PollPendingFrees();

    size_t best_idx = free_list_.size();
    size_t best_waste = SIZE_MAX;

    for (size_t i = 0; i < free_list_.size(); ++i) {
      const Block& block = free_list_[i];
      if (block.size >= alloc_size) {
        size_t waste = block.size - alloc_size;
        if (waste < best_waste) {
          best_waste = waste;
          best_idx = i;
          if (waste == 0) break;
        }
      }
    }

    if (best_idx < free_list_.size()) {
      Block block = free_list_[best_idx];
      if (best_idx != free_list_.size() - 1) {
        free_list_[best_idx] = free_list_.back();
      }
      free_list_.pop_back();

      VLOG(2) << "PinnedMemoryPool: Reusing block " << block.ptr
              << " size=" << block.size << " requested=" << alloc_size;
      return block.ptr;
    }
  }

  // No suitable block found, allocate new memory
  musaSetDevice(device_id_);

  void* ptr = nullptr;
  musaError_t err = musaHostAlloc(&ptr, alloc_size, musaHostAllocDefault);

  if (err != musaSuccess) {
    LOG(WARNING) << "PinnedMemoryPool: musaHostAlloc failed for " << alloc_size
                 << " bytes: " << musaGetErrorString(err);
    return nullptr;
  }

  VLOG(2) << "PinnedMemoryPool: Allocated new block " << ptr
          << " size=" << alloc_size;
  return ptr;
}

void GPUPinnedMemoryPool::FreeAsync(void* ptr, size_t bytes,
                                    musaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

  size_t alloc_size = bytes;
  if (alloc_size < kMinAllocationSize) {
    alloc_size = kMinAllocationSize;
  }

  musaSetDevice(device_id_);

  // If stream is nullptr, synchronize and free immediately
  // This avoids creating unnecessary events and ensures memory is safe to free
  if (stream == nullptr) {
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
      LOG(WARNING) << "PinnedMemoryPool: musaDeviceSynchronize failed: "
                   << musaGetErrorString(err);
    }
    musaError_t free_err = musaFreeHost(ptr);
    if (free_err != musaSuccess) {
      LOG(ERROR) << "PinnedMemoryPool: musaFreeHost failed: "
                 << musaGetErrorString(free_err);
    }
    VLOG(2) << "PinnedMemoryPool: Freed block " << ptr << " size=" << alloc_size
            << " (sync, no stream)";
    return;
  }

  musaEvent_t event;
  musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
  if (err != musaSuccess) {
    LOG(ERROR) << "PinnedMemoryPool: Failed to create event: "
               << musaGetErrorString(err);
    musaStreamSynchronize(stream);
    musaFreeHost(ptr);
    return;
  }

  err = musaEventRecord(event, stream);
  if (err != musaSuccess) {
    LOG(ERROR) << "PinnedMemoryPool: Failed to record event: "
               << musaGetErrorString(err);
    musaEventDestroy(event);
    musaStreamSynchronize(stream);
    musaFreeHost(ptr);
    return;
  }

  {
    mutex_lock l(mu_);
    pending_frees_.push_back({ptr, alloc_size, event});
    poll_cv_.notify_one();
  }

  VLOG(2) << "PinnedMemoryPool: FreeAsync block " << ptr
          << " size=" << alloc_size << " stream=" << stream;
}

void GPUPinnedMemoryPool::PollLoop() {
  musaSetDevice(device_id_);

  while (true) {
    {
      mutex_lock l(mu_);

      while (!stop_polling_ && pending_frees_.empty()) {
        poll_cv_.wait(l);
      }

      if (stop_polling_) {
        PollPendingFrees();
        break;
      }

      PollPendingFrees();
    }

    Env::Default()->SleepForMicroseconds(kPollingIntervalUs);
  }
}

void GPUPinnedMemoryPool::PollPendingFrees() {
  auto it = pending_frees_.begin();
  while (it != pending_frees_.end()) {
    Block& block = *it;

    // Skip blocks with null events (should not happen, but be defensive)
    if (block.event == nullptr) {
      LOG(WARNING) << "PinnedMemoryPool: Block " << block.ptr
                   << " has null event, moving to free_list";
      free_list_.push_back(block);
      it = pending_frees_.erase(it);
      continue;
    }

    musaError_t err = musaEventQuery(block.event);

    if (err == musaSuccess) {
      // Event completed successfully, destroy it and move block to free_list
      musaError_t destroy_err = musaEventDestroy(block.event);
      if (destroy_err != musaSuccess) {
        LOG(WARNING) << "PinnedMemoryPool: Failed to destroy event for block "
                     << block.ptr << ": " << musaGetErrorString(destroy_err);
      }
      block.event = nullptr;

      VLOG(2) << "PinnedMemoryPool: Block " << block.ptr
              << " completed, moving to free_list";

      free_list_.push_back(block);
      it = pending_frees_.erase(it);
    } else if (err == musaErrorNotReady) {
      // Event not ready yet, skip to next block
      ++it;
    } else {
      // Event query failed, destroy the event and move block to free_list
      // This is a defensive measure to prevent memory leaks
      LOG(WARNING) << "PinnedMemoryPool: Event query failed for block "
                   << block.ptr << ": " << musaGetErrorString(err);
      musaError_t destroy_err = musaEventDestroy(block.event);
      if (destroy_err != musaSuccess) {
        LOG(WARNING) << "PinnedMemoryPool: Failed to destroy event for block "
                     << block.ptr << ": " << musaGetErrorString(destroy_err);
      }
      block.event = nullptr;
      free_list_.push_back(block);
      it = pending_frees_.erase(it);
    }
  }
}

void GPUPinnedMemoryPool::ReleaseBlock(const Block& block) {
  if (block.ptr != nullptr) {
    musaError_t err = musaFreeHost(block.ptr);
    if (err != musaSuccess) {
      LOG(WARNING) << "PinnedMemoryPool: musaFreeHost failed for " << block.ptr
                   << ": " << musaGetErrorString(err);
    } else {
      VLOG(2) << "PinnedMemoryPool: Released block " << block.ptr;
    }
  }
}

}  // namespace musa
}  // namespace tensorflow
