#ifndef MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMCPY_H_
#define MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMCPY_H_

#include <mudnn.h>
#include <musa_runtime_api.h>

#include "kernel_register.h"

using mTensor = ::musa::dnn::Tensor;
using mHandle = ::musa::dnn::Handle;
using mStatus = ::musa::dnn::Status;

namespace tensorflow {
namespace musa {

mStatus MusaMemcpyD2H(void* h, const void* d, size_t btype);
mStatus MusaMemcpyH2D(void* d, const void* h, size_t btype);
mStatus MusaMemcpyD2D(void* d1, const void* d2, size_t btype);

mStatus MusaMemcpyAsyncD2H(void* h, const void* d, size_t btype,
                           musaStream_t s = 0);
mStatus MusaMemcpyAsyncH2D(void* d, const void* h, size_t btype,
                           musaStream_t s = 0);
mStatus MusaMemcpyAsyncD2D(void* d1, const void* d2, size_t btype,
                           musaStream_t s = 0);

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMCPY_H_
