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
