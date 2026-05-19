#ifndef TENSORFLOW_MUSA_TF_STATUS_COMPAT_H_
#define TENSORFLOW_MUSA_TF_STATUS_COMPAT_H_

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

#if TF_MAJOR_VERSION < 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 10)
inline Status OkStatus() { return Status::OK(); }
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_TF_STATUS_COMPAT_H_
