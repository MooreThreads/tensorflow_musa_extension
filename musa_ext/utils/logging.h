#ifndef MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
#define MUSA_PLUGIN_SRC_UTILS_LOGGING_H_

#include <mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

#ifndef NDEBUG
#define DLOG LOG
#else
#define DLOG(severity) \
  while (false) ::tensorflow::internal::LogMessageNull()
#endif

#define MUSA_CHECK_LOG(status, msg)              \
  if (status != musaSuccess) {                   \
    LOG(ERROR) << "[MUSA ERROR] " << msg << ": " \
               << musaGetErrorString(status);    \
    return ::musa::dnn::Status::INTERNAL_ERROR;  \
  }

// Note: MTOP_CHECK_LOG, MTOP_CHECK_OK, MTOP_CHECK_OK_RUN, and
// MTOP_CHECK_MTDNN_STATUS_RET are defined in kernels/utils_op.h
// Use those for consistency across the codebase

// Legacy kernel timing macros removed.


#endif  // MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
