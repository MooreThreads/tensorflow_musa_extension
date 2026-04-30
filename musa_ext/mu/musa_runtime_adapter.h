#ifndef TENSORFLOW_MUSA_MU_MUSA_RUNTIME_ADAPTER_H_
#define TENSORFLOW_MUSA_MU_MUSA_RUNTIME_ADAPTER_H_

#include <cstdint>

#include <musa_runtime.h>

#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace musa {
namespace runtime {

// Propagate a musa runtime error to TF_Status.
void SetStatusFromMusa(TF_Status* status, musaError_t err, const char* context);

// Current device ordinal from SP_Device (expects device_handle to be set in
// create_device, or falls back to SP_Device::ordinal).
int GetDeviceOrdinal(const void* device_handle, int32_t ordinal_fallback);

// musaSetDevice for the given ordinal; records error in status on failure.
void SetMusaDeviceOrStatus(int device_ordinal, TF_Status* status);

}  // namespace runtime
}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_MUSA_RUNTIME_ADAPTER_H_
