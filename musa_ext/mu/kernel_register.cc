// C kernel registration hook for TensorFlow Pluggable / modular kernel paths.
//
// Current MUSA op kernels are registered with C++ static initialization via
// REGISTER_KERNEL_BUILDER in individual `*.cc` files; that runs when
// `libmusa_plugin.so` is loaded, without calling `TF_InitKernel()`.
//
// The vector below is only populated if code uses the `MUSA_KERNEL_REGISTER`
// macro in `kernel_register.h` to defer registration to an explicit
// `TF_InitKernel()` call. Reserved for future C-API-based kernel registration.

#include "kernel_register.h"

#include <algorithm>
#include <vector>

#include "tensorflow/c/kernels.h"

namespace {
std::vector<::tensorflow::musa::RegFuncPtr> RegVector;
}

void TF_InitKernel() {
  std::for_each(RegVector.cbegin(), RegVector.cend(),
                [](void (*const regFunc)()) { (*regFunc)(); });
}

namespace tensorflow {
namespace musa {

bool musaKernelRegFunc(RegFuncPtr regFunc) {
  RegVector.push_back(regFunc);
  return true;
}

}  // namespace musa
}  // namespace tensorflow
