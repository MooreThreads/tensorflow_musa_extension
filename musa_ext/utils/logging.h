#ifndef MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
#define MUSA_PLUGIN_SRC_UTILS_LOGGING_H_

#include <mudnn.h>
#include <musa_runtime.h>

#include <string>
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
// Use those for consistency across the codebase.

// -----------------------------------------------------------------------------
// Kernel timing instrumentation macros (compiled out)
// -----------------------------------------------------------------------------
//
// Historically this header shipped a musaEvent-based per-kernel timing
// scope gated behind the `MUSA_KERNEL_DEBUG` compile flag. That path was
// incomplete, duplicated device-side timing TF already offers via Grappler
// traces, and added ~760 LOC plus a runtime env-var surface that drifted out
// of sync with the rest of the allocator work, so the implementation has
// been removed.
//
// The macros and empty `KernelTimingStageSpec`/`KernelTimingLayout` types
// are kept here so existing kernel call sites
// (`MUSA_KERNEL_TIMING_GUARD(ctx)`, `MUSA_KERNEL_TRACE_START("foo")`, etc.)
// still compile unchanged. They intentionally expand to no-ops — if we ever
// re-introduce per-kernel timing, the call sites won't need to be touched
// again.

namespace tensorflow {
namespace musa {
namespace timing {

struct KernelTimingStageSpec {
  KernelTimingStageSpec(const std::string& /*id*/, const std::string& /*name*/,
                        bool /*show_zero_ms*/ = false) {}
};

using KernelTimingLayout = std::vector<KernelTimingStageSpec>;

}  // namespace timing
}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_TIMING_STAGE(stage_id, display_name, show_zero) \
  ::tensorflow::musa::timing::KernelTimingStageSpec(                \
      (stage_id), (display_name), (show_zero))

#define MUSA_KERNEL_TIMING_LAYOUT(...) \
  ::tensorflow::musa::timing::KernelTimingLayout { __VA_ARGS__ }

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME_AND_LAYOUT(ctx, kernel_name, \
                                                      layout)           \
  do {                                                                  \
  } while (false)

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, kernel_name) \
  do {                                                       \
  } while (false)

#define MUSA_KERNEL_TIMING_GUARD_WITH_LAYOUT(ctx, layout) \
  do {                                                    \
  } while (false)

#define MUSA_KERNEL_TIMING_GUARD(ctx) \
  do {                                \
  } while (false)

#define MUSA_KERNEL_TRACE_START(stage_name) \
  do {                                      \
  } while (false)

#define MUSA_KERNEL_TRACE_END(stage_name) \
  do {                                    \
  } while (false)

#define MUSA_KERNEL_TRACE(stage_name) \
  do {                                \
  } while (false)

#endif  // MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
