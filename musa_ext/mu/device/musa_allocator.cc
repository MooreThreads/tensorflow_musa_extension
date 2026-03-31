#include "musa_allocator.h"

#include <algorithm>
#include <cstring>

namespace tensorflow {
namespace musa {

// MusaSubAllocator implementation is header-only inline
// This file is kept for future allocator extensions

// Memory Coloring Runtime Control Functions
// These can be called from application code to enable/disable debugging features

void EnableMemoryDebugging(bool enable_coloring, bool enable_history,
                           bool enable_verification) {
  EnableMemoryColoring(enable_coloring);
  EnableMemoryHistoryTracking(enable_history);
  EnableMemoryVerification(enable_verification);

  LOG(INFO) << "[MemoryForensics] Memory debugging configured: "
            << "coloring=" << enable_coloring
            << " history=" << enable_history
            << " verification=" << enable_verification;
}

void DisableMemoryDebugging() {
  EnableMemoryColoring(false);
  EnableMemoryHistoryTracking(false);
  EnableMemoryVerification(false);

  LOG(INFO) << "[MemoryForensics] Memory debugging disabled";
}

MemoryStats GetAllocatorMemoryStats() {
  return GetMemoryStats();
}

std::string DumpMemoryForensicsReport(void* ptr) {
  return GetMemoryForensicsReport(ptr);
}

}  // namespace musa
}  // namespace tensorflow