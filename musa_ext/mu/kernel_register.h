#ifndef TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
#define TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_

// Optional deferred kernel registration. Most MUSA kernels use
// `REGISTER_KERNEL_BUILDER` in their translation units. Use
// `MUSA_KERNEL_REGISTER` only when you need registration to run from
// `TF_InitKernel()` (e.g. C API kernel builders).

#include "../kernels/utils_op.h"
#include "./device_register.h"

namespace tensorflow {
namespace musa {

typedef void (*RegFuncPtr)();

bool musaKernelRegFunc(RegFuncPtr regFunc);

class MusaAnnotatedTraceMe {
 public:
  template <typename... Args>
  explicit MusaAnnotatedTraceMe(Args&&... args) {}
};

// Note: MTOP_CHECK_OK and MTOP_CHECK_OK_RUN are defined in utils_op.h
// Use those macros for consistency across the codebase

}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_REGISTER(name)                                 \
  static void musaKernelReg_##name();                              \
  static bool musa_kernel_registered_##name =                      \
      ::tensorflow::musa::musaKernelRegFunc(musaKernelReg_##name); \
  static void musaKernelReg_##name()

#endif  // TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
