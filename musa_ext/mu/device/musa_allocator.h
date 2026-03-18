#ifndef TENSORFLOW_MUSA_ALLOCATOR_H_
#define TENSORFLOW_MUSA_ALLOCATOR_H_

#include <musa_runtime.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace musa {

// MusaSubAllocator wraps musaMalloc/musaFree for use with TensorFlow's
// BFCAllocator. This replaces direct musaMalloc calls with a proper memory
// pooling strategy.
class MusaSubAllocator : public SubAllocator {
 public:
  MusaSubAllocator(int device_id, const std::vector<Visitor>& alloc_visitors,
                   const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors), device_id_(device_id) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    if (num_bytes == 0) {
      *bytes_received = 0;
      return nullptr;
    }

    // Ensure minimum alignment of 256 bytes (musaMalloc guarantee)
    // and respect the requested alignment from BFCAllocator
    size_t min_alignment = 256;
    if (alignment < min_alignment) {
      alignment = min_alignment;
    }

    // Round up allocation size to alignment boundary
    size_t alloc_size = (num_bytes + alignment - 1) & ~(alignment - 1);
    if (alloc_size < num_bytes) {
      // Overflow check
      return nullptr;
    }

    void* ptr = nullptr;
    musaSetDevice(device_id_);
    musaError_t err = musaMalloc(&ptr, alloc_size);
    if (err != musaSuccess) {
      LOG(WARNING) << "MusaSubAllocator: musaMalloc failed for " << alloc_size
                   << " bytes (alignment=" << alignment
                   << "): " << musaGetErrorString(err);
      return nullptr;
    }

    // Check alignment
    if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
      LOG(WARNING) << "MusaSubAllocator: musaMalloc returned unaligned pointer "
                   << ptr << " (requested alignment=" << alignment << ")";
      musaFree(ptr);
      return nullptr;
    }

    *bytes_received = alloc_size;

    // Call visitor to track allocation
    VisitAlloc(ptr, device_id_, alloc_size);

    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      // Call visitor to track deallocation
      VisitFree(ptr, device_id_, num_bytes);

      musaSetDevice(device_id_);
      musaError_t err = musaFree(ptr);
      if (err != musaSuccess) {
        LOG(ERROR) << "MusaSubAllocator: musaFree failed: "
                   << musaGetErrorString(err);
      }
    }
  }

  bool SupportsCoalescing() const override { return true; }

 private:
  int device_id_;
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_ALLOCATOR_H_
