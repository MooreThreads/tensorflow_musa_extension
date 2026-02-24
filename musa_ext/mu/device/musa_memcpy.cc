#include "mu/device/musa_memcpy.h"

#include <musa_runtime.h>
#include <stdio.h>

namespace tensorflow {
namespace musa {

mStatus MusaMemcpyD2H(void* h, const void* d, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }

  musaError_t err = musaMemcpy(h, d, size, musaMemcpyDeviceToHost);

  if (err != musaSuccess) {
    return static_cast<mStatus>(1);
  }

  return mStatus::SUCCESS;
}

mStatus MusaMemcpyH2D(void* d, const void* h, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpy(d, h, size, musaMemcpyHostToDevice);

  if (err != musaSuccess) {
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyD2D(void* d1, const void* d2, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpy(d1, d2, size, musaMemcpyDeviceToDevice);

  if (err != musaSuccess) {
    return static_cast<mStatus>(1);
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
    return static_cast<mStatus>(1);
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
    return static_cast<mStatus>(1);
  }

  musaError_t err = musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, s);

  if (err != musaSuccess) {
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

}  // namespace musa
}  // namespace tensorflow
