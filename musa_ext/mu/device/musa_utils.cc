#include "musa_utils.h"

#include "musa_device.h"

namespace tensorflow {
namespace musa {

musaStream_t GetMusaStream(MusaDevice* device) {
  if (!device) return nullptr;

  return device->GetStream();
}

void MusaSyncAllActivity(int device_id) {
  musaSetDevice(device_id);
  musaDeviceSynchronize();
}

}  // namespace musa
}  // namespace tensorflow
