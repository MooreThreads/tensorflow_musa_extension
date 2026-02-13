#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_UTILS_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_UTILS_H_

#include <musa_runtime.h>

#include <cstdint>

namespace tensorflow {
namespace musa {

class MusaDevice;

void* MusaAllocateAligned(MusaDevice* device, uint64_t size);

musaStream_t GetMusaStream(MusaDevice* device);

void MusaSyncAllActivity(int device_id);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU1_DEVICE_MUSA_UTILS_H_
