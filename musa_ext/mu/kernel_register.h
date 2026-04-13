#ifndef TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
#define TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_

#include "../kernels/utils_op.h"
#include "./device_register.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace musa {

typedef void (*RegFuncPtr)();

bool musaKernelRegFunc(RegFuncPtr regFunc);

// Note: MTOP_CHECK_OK and MTOP_CHECK_OK_RUN are defined in utils_op.h.
// Use those macros for consistency across the codebase.

}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_REGISTER(name)                                 \
  static void musaKernelReg_##name();                              \
  static bool musa_kernel_registered_##name =                      \
      ::tensorflow::musa::musaKernelRegFunc(musaKernelReg_##name); \
  static void musaKernelReg_##name()

#endif  // TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
