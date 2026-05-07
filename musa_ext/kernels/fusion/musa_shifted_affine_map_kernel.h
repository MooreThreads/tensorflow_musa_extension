#ifndef TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_SHIFTED_AFFINE_MAP_KERNEL_H_
#define TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_SHIFTED_AFFINE_MAP_KERNEL_H_

#include <musa_runtime.h>

#include <cstdint>

namespace tensorflow {
namespace musa {

constexpr int kShiftedAffineMapMaxDims = 8;

struct ShiftedAffineMapShape {
  int rank;
  int dims[kShiftedAffineMapMaxDims];
};

struct ShiftedAffineMapStrides {
  int values[kShiftedAffineMapMaxDims];
};

template <typename T>
void LaunchShiftedAffineMapKernel(
    const T* data_left, ShiftedAffineMapStrides data_left_st, const T* mask,
    ShiftedAffineMapStrides mask_st, const T* sliced_var_right,
    ShiftedAffineMapStrides sliced_var_right_st, T* output,
    ShiftedAffineMapShape shape, int total_elements, musaStream_t stream);

template <typename T>
void LaunchShiftedAffineMapContiguous(const T* data_left, const T* mask,
                                      const T* sliced_var_right, T* output,
                                      int64_t total_elements,
                                      musaStream_t stream);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_EXTENSION_KERNELS_MATH_MUSA_SHIFTED_AFFINE_MAP_KERNEL_H_
