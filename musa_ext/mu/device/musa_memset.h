#ifndef MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMSET_H_
#define MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMSET_H_

#include <mudnn.h>

namespace tensorflow {
namespace musa {

using mTensor = ::musa::dnn::Tensor;
using mHandle = ::musa::dnn::Handle;
using mStatus = ::musa::dnn::Status;

mStatus Memset(mHandle& h, void* device_dst, uint64_t size, uint8_t pattern);

mStatus Memset32(mHandle& h, void* device_dst, uint64_t size, uint32_t pattern);

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_MU_DEVICE_MUSA_MEMSET_H_
