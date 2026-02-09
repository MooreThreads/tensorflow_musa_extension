#ifndef MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
#define MUSA_PLUGIN_SRC_UTILS_LOGGING_H_

#include "tensorflow/core/platform/logging.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <iostream>

#ifndef NDEBUG
#define DLOG LOG
#else
#define DLOG(severity) \
  while (false)        \
  ::tensorflow::internal::LogMessageNull()
#endif

#define MUSA_CHECK_LOG(status, msg) \
  if (status != musaSuccess) { \
     LOG(ERROR) << "[MUSA ERROR] " << msg << ": " << musaGetErrorString(status); \
     return ::musa::dnn::Status::INTERNAL_ERROR; \
  }

#define MTOP_CHECK_MTDNN_STATUS_RET(statement) \
  { \
    ::musa::dnn::Status _status = (statement); \
    if (_status != ::musa::dnn::Status::SUCCESS) { \
       LOG(ERROR) << "[MUDNN ERROR] Status: " << static_cast<int>(_status); \
       return _status; \
    } \
  }

#endif  // MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
