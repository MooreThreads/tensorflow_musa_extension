#include "mu/musa_runtime_adapter.h"

#include <cstdio>
#include <cstring>

namespace tensorflow {
namespace musa {
namespace runtime {

void SetStatusFromMusa(TF_Status* status, musaError_t err, const char* context) {
  if (status == nullptr) return;
  if (err == musaSuccess) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  char buf[512];
  snprintf(buf, sizeof(buf), "%s: %s", context ? context : "musa",
           musaGetErrorString(err));
  TF_SetStatus(status, TF_INTERNAL, buf);
}

int GetDeviceOrdinal(const void* device_handle, int32_t ordinal_fallback) {
  if (device_handle != nullptr) {
    return *static_cast<const int*>(device_handle);
  }
  return static_cast<int>(ordinal_fallback);
}

void SetMusaDeviceOrStatus(int device_ordinal, TF_Status* status) {
  musaError_t err = musaSetDevice(device_ordinal);
  SetStatusFromMusa(status, err, "musaSetDevice");
}

}  // namespace runtime
}  // namespace musa
}  // namespace tensorflow
