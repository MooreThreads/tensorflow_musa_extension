#ifndef TENSORFLOW_MUSA_ALLOCATOR_H_
#define TENSORFLOW_MUSA_ALLOCATOR_H_

#include <musa_runtime.h>

#include <algorithm>
#include <limits>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {
namespace musa {

class MusaRawAllocator : public Allocator {
 public:
  explicit MusaRawAllocator(int device_id) : device_id_(device_id) {}

  ~MusaRawAllocator() override = default;

  std::string Name() override { return "musa_raw_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes == 0) return nullptr;

    musaSetDevice(device_id_);

    size_t target_alignment = std::max((size_t)256, alignment);

    // Check for overflow before calculation
    if (num_bytes > std::numeric_limits<size_t>::max() - target_alignment) {
      LOG(ERROR) << "MUSA allocator: allocation size overflow: " << num_bytes;
      return nullptr;
    }

    size_t alloc_bytes = (num_bytes + target_alignment - 1) / target_alignment *
                         target_alignment;

    // Check for overflow after adding padding
    if (alloc_bytes > std::numeric_limits<size_t>::max() - 256) {
      LOG(ERROR) << "MUSA allocator: allocation size overflow after padding: "
                 << alloc_bytes;
      return nullptr;
    }
    alloc_bytes += 256;

    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, alloc_bytes);
    if (err != musaSuccess) {
      LOG(ERROR) << "MUSA allocator: musaMalloc failed: "
                 << musaGetErrorString(err) << " size: " << alloc_bytes;
      return nullptr;
    }
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr) {
      musaSetDevice(device_id_);
      musaError_t err = musaFree(ptr);
      if (err != musaSuccess) {
        LOG(ERROR) << "MUSA allocator: musaFree failed: "
                   << musaGetErrorString(err);
      }
    }
  }

 private:
  int device_id_;
};

}  // namespace musa
}  // namespace tensorflow
#endif
