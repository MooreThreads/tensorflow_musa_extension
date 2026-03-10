#include "mu/device/musa_memcpy.h"

#include <musa_runtime.h>
#include <stdio.h>

namespace tensorflow {
namespace musa {

// For framework-level operations, TensorFlow's stream dependency tracking
// should handle synchronization naturally. Synchronous versions should only
// be used when absolutely necessary (e.g., when CPU needs to access data immediately).

mStatus MusaMemcpyD2H(void* h, const void* d, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (h == nullptr || d == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyD2H failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", h, d, size);
    return static_cast<mStatus>(1);
  }

  // Use true asynchronous copy without immediate synchronization.
  // The caller is responsible for synchronization when the data is actually needed.
  // This allows overlapping computation with data transfer.
  //
  // Note: Using default stream (0) for compatibility. For true async behavior,
  // use MusaMemcpyAsyncD2H with a non-blocking stream.
  musaError_t err = musaMemcpyAsync(h, d, size, musaMemcpyDeviceToHost, 0);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsync D2H failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), h, d, size);
    return static_cast<mStatus>(1);
  }


  return mStatus::SUCCESS;
}

mStatus MusaMemcpyH2D(void* d, const void* h, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyH2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", d, h, size);
    return static_cast<mStatus>(1);
  }

  // Use true asynchronous copy without immediate synchronization.
  // H2D transfers can overlap with computation on the device.
  musaError_t err = musaMemcpyAsync(d, h, size, musaMemcpyHostToDevice, 0);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsync H2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d, h, size);
    return static_cast<mStatus>(1);
  }


  return mStatus::SUCCESS;
}

mStatus MusaMemcpyD2D(void* d1, const void* d2, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyD2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", d1, d2, size);
    return static_cast<mStatus>(1);
  }

  //  D2D transfers on the same device don't need synchronization
  // as they execute on the device's copy engine. Removing sync allows overlapping
  // with compute kernels.
  musaError_t err = musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, 0);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsync D2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d1, d2, size);
    return static_cast<mStatus>(1);
  }

  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2H(void* h, const void* d, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (h == nullptr || d == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncD2H failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", h, d, size);
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpyAsync(h, d, size, musaMemcpyDeviceToHost, s);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncD2H failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), h, d, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncH2D(void* d, const void* h, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncH2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", d, h, size);
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpyAsync(d, h, size, musaMemcpyHostToDevice, s);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncH2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d, h, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2D(void* d1, const void* d2, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncD2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n", d1, d2, size);
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, s);

  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyAsyncD2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d1, d2, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

}  // namespace musa
}  // namespace tensorflow
