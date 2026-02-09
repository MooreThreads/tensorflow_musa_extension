#ifndef TENSORFLOW_MUSA_ALLOCATOR_H_
#define TENSORFLOW_MUSA_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include <musa_runtime.h>
#include <string>
#include <algorithm> 

namespace tensorflow {
namespace musa {

class MusaRawAllocator : public Allocator {
 public:
  // 构造函数
  explicit MusaRawAllocator(int device_id) : device_id_(device_id) {}
  
  ~MusaRawAllocator() override = default;

  std::string Name() override { return "musa_raw_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes == 0) return nullptr;

    // 切换到分配器绑定的物理卡
    musaSetDevice(device_id_);

    size_t target_alignment = std::max((size_t)256, alignment);
    size_t alloc_bytes = (num_bytes + target_alignment - 1) / target_alignment * target_alignment;
    alloc_bytes += 256; 

    void* ptr = nullptr;
    if (musaMalloc(&ptr, alloc_bytes) != musaSuccess) {
        return nullptr;
    }
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr) {
      // 释放时也需要切换上下文
      musaSetDevice(device_id_);
      musaFree(ptr);
    }
  }

 private:
  // 定义成员变量
  int device_id_; 
};

} // namespace musa
} // namespace tensorflow
#endif
