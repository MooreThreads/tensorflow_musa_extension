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

#ifndef TENSORFLOW_MUSA_PINNED_MEMORY_POOL_H_
#define TENSORFLOW_MUSA_PINNED_MEMORY_POOL_H_

#include <musa_runtime.h>

#include <atomic>
#include <list>
#include <thread>
#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace musa {

// GPUPinnedMemoryPool implements a GPU-aware memory pool for Pinned Host
// Memory.
//
// Problem: BFCAllocator cannot be used for bounce buffers because it reuses
// memory addresses immediately after DeallocateRaw(), but GPU async copies
// may still be in progress.
//
// Solution: This pool tracks GPU event completion before reusing memory.
// - Allocate() returns memory from free_list_ or allocates new pinned memory
// - FreeAsync() records an event and moves block to pending_frees_
// - Polling thread periodically checks events, moves completed blocks to
// free_list_
//
// This ensures memory is never reused while GPU async copies are in progress.
class GPUPinnedMemoryPool {
 public:
  explicit GPUPinnedMemoryPool(int device_id);
  ~GPUPinnedMemoryPool();

  // Allocate pinned memory. Returns nullptr on failure.
  // Thread-safe.
  void* Allocate(size_t bytes);

  // Free pinned memory asynchronously.
  // The memory is not immediately reused; it's returned to free_list_ only
  // after the GPU stream completes all prior operations.
  // Thread-safe.
  void FreeAsync(void* ptr, size_t bytes, musaStream_t stream);

  // Disable copy and move
  GPUPinnedMemoryPool(const GPUPinnedMemoryPool&) = delete;
  GPUPinnedMemoryPool& operator=(const GPUPinnedMemoryPool&) = delete;

 private:
  struct Block {
    void* ptr;
    size_t size;
    musaEvent_t event;
  };

  const int device_id_;

  mutex mu_;

  // Free blocks that are safe to reuse (GPU operations completed)
  std::vector<Block> free_list_ TF_GUARDED_BY(mu_);

  // Blocks waiting for GPU completion
  std::list<Block> pending_frees_ TF_GUARDED_BY(mu_);

  // Background polling thread
  std::thread polling_thread_;
  std::atomic<bool> stop_polling_;
  condition_variable poll_cv_ TF_GUARDED_BY(mu_);

  // Background loop that polls pending frees
  void PollLoop();

  // Check pending_frees_ and move completed blocks to free_list_
  // Must hold mu_ when calling
  void PollPendingFrees() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Actually free a block (call musaFreeHost)
  static void ReleaseBlock(const Block& block);
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_PINNED_MEMORY_POOL_H_
