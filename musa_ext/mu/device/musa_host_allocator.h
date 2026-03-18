#ifndef TENSORFLOW_MUSA_HOST_ALLOCATOR_H_
#define TENSORFLOW_MUSA_HOST_ALLOCATOR_H_

#include <musa_runtime.h>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace musa {

// SubAllocator for host pinned memory using musaHostAlloc/musaFreeHost.
// Used by BFCAllocator for host memory allocation.
class MusaHostSubAllocator : public SubAllocator {
 public:
  MusaHostSubAllocator(const std::vector<Visitor>& alloc_visitors,
                       const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    if (num_bytes == 0) {
      *bytes_received = 0;
      return nullptr;
    }

    size_t min_alignment = 256;
    if (alignment < min_alignment) {
      alignment = min_alignment;
    }

    size_t alloc_size = (num_bytes + alignment - 1) & ~(alignment - 1);
    if (alloc_size < num_bytes) {
      return nullptr;
    }

    void* ptr = nullptr;
    musaError_t err = musaHostAlloc(&ptr, alloc_size, musaHostAllocDefault);
    if (err != musaSuccess) {
      LOG(ERROR) << "MusaHostSubAllocator: musaHostAlloc failed for "
                 << alloc_size << " bytes: " << musaGetErrorString(err);
      return nullptr;
    }

    if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
      LOG(WARNING)
          << "MusaHostSubAllocator: musaHostAlloc returned unaligned pointer "
          << ptr << " (requested alignment=" << alignment << ")";
      musaFreeHost(ptr);
      return nullptr;
    }

    *bytes_received = alloc_size;
    VisitAlloc(ptr, 0, alloc_size);

    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      VisitFree(ptr, 0, num_bytes);

      musaError_t err = musaFreeHost(ptr);
      if (err != musaSuccess) {
        LOG(ERROR) << "MusaHostSubAllocator: musaFreeHost failed: "
                   << musaGetErrorString(err);
      }
    }
  }

  bool SupportsCoalescing() const override { return true; }
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_HOST_ALLOCATOR_H_
