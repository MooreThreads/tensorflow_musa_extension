#include "mu/device/musa_memcpy.h"

#include <musa_runtime.h>
#include <stdio.h>

namespace tensorflow {
namespace musa {

mStatus MusaMemcpyD2H(void* h, const void* d, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }

  // fprintf(stderr, "\n[MUSA_ADDR_TRACE] D2H Attempt:\n");
  // fprintf(stderr, "  - Size: %zu bytes\n", size);
  // fprintf(stderr, "  - Src Ptr (d): %p\n", d);
  // fprintf(stderr, "  - Dst Ptr (h): %p\n", h);

  musaError_t err = musaMemcpy(h, d, size, musaMemcpyDeviceToHost);

  if (err != musaSuccess) {
    // fprintf(stderr, "  - [RESULT] FAILED: %s\n", musaGetErrorString(err));
    return static_cast<mStatus>(1);
  }

  // fprintf(stderr, "  - [RESULT] SUCCESS\n");
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyH2D(void* d, const void* h, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpy(d, h, size, musaMemcpyHostToDevice);

  if (err != musaSuccess) {
    // fprintf(stderr, ">>> [MUSA] H2D Sync Failed: %s, size: %zu, src: %p, dst:
    // %p\n",
    //         musaGetErrorString(err), size, h, d);
    // return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyD2D(void* d1, const void* d2, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpy(d1, d2, size, musaMemcpyDeviceToDevice);

  if (err != musaSuccess) {
    // fprintf(stderr, ">>> [MUSA] D2D Sync Failed: %s, size: %zu\n",
    //         musaGetErrorString(err), size);
    // return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2H(void* h, const void* d, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpyAsync(h, d, size, musaMemcpyDeviceToHost, s);

  if (err != musaSuccess) {
    // fprintf(stderr, ">>> [MUSA] D2H Async Failed: %s, size: %zu, stream:
    // %p\n",
    //         musaGetErrorString(err), size, s);
    // return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncH2D(void* d, const void* h, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpyAsync(d, h, size, musaMemcpyHostToDevice, s);

  if (err != musaSuccess) {
    // fprintf(stderr, ">>> [MUSA] H2D Async Failed: %s, size: %zu, stream:
    // %p\n",
    //         musaGetErrorString(err), size, s);
    // return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2D(void* d1, const void* d2, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, s);

  if (err != musaSuccess) {
    // fprintf(stderr, ">>> [MUSA] D2D Async Failed: %s, size: %zu, stream:
    // %p\n",
    //         musaGetErrorString(err), size, s);
    // return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

}  // namespace musa
}  // namespace tensorflow
