#ifndef MUSA_PLUGIN_SRC_KERNELS_MUSA_STRIDE_INFLATE_KERNEL_H_
#define MUSA_PLUGIN_SRC_KERNELS_MUSA_STRIDE_INFLATE_KERNEL_H_

#include <musa_runtime.h>

#include <cstdint>

namespace tensorflow {
namespace musa {

constexpr int kMaxStrideInflateDims = 8;

struct DimSizeArray {
  int64_t value[kMaxStrideInflateDims];
};

template <typename T>
void MusaStrideKernelLauncher(musaStream_t stream, int64_t size, const T* in,
                              T* out, DimSizeArray dims, DimSizeArray strides,
                              int ndims);

template <typename T>
void MusaInflateKernelLauncher(musaStream_t stream, int64_t input_size,
                               const T* in, T* out, DimSizeArray input_dims,
                               DimSizeArray strides, int ndims,
                               int64_t output_size);

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_MUSA_STRIDE_INFLATE_KERNEL_H_
