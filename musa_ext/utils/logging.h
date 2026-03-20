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

#ifdef MUSA_KERNEL_DEBUG
namespace tensorflow {
namespace musa {
// Defined in kernels/utils_op.h.
musaStream_t GetMusaStreamByCtx(tensorflow::OpKernelContext* context);

namespace timing {

struct KernelTimingConfig {
  int level = 0;
  bool stats = false;
  std::string kernel_filter = "ALL";
  bool enabled = false;
};

struct KernelTimingStats {
  std::string kernel_name;
  std::string input_shape;
  uint64_t count = 0;
  double total_ms = 0.0;
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = 0.0;
};

struct KernelTimingStageSpec {
  std::string stage_id;
  std::string display_name;
  bool show_zero = false;

  KernelTimingStageSpec(const std::string& id, const std::string& name,
                        bool show_zero_ms = false)
      : stage_id(id), display_name(name), show_zero(show_zero_ms) {}
};

using KernelTimingLayout = std::vector<KernelTimingStageSpec>;

inline std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

inline int ReadEnvInt(const char* key, int default_value) {
  const char* value = std::getenv(key);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::atoi(value);
}

inline const KernelTimingConfig& GetKernelTimingConfig() {
  static const KernelTimingConfig* config = []() {
    auto* c = new KernelTimingConfig();
    c->level = ReadEnvInt(
        "MUSA_TIMING_KERNEL_LEVEL", ReadEnvInt("MUSA_KERNEL_LEVEL", 0));
    c->stats = ReadEnvInt("MUSA_TIMING_KERNEL_STATS",
                          ReadEnvInt("MUSA_KERNEL_STATS", 0)) == 1;

    const char* kernel_name = std::getenv("MUSA_TIMING_KERNEL_NAME");
    if (kernel_name == nullptr || kernel_name[0] == '\0') {
      kernel_name = std::getenv("MUSA_KERNEL_NAME");
    }
    if (kernel_name != nullptr && kernel_name[0] != '\0') {
      c->kernel_filter = kernel_name;
    } else {
      c->kernel_filter = "ALL";
    }

    c->enabled = (c->level >= 1);
    return c;
  }();
  return *config;
}

inline bool ShouldTraceKernelName(const std::string& kernel_name) {
  const auto& cfg = GetKernelTimingConfig();
  if (!cfg.enabled) return false;

  const std::string filter = ToLower(cfg.kernel_filter);
  if (filter == "all") return true;

  const std::string current = ToLower(kernel_name);
  return current.find(filter) != std::string::npos;
}

inline std::string BuildInputShapeSummary(OpKernelContext* ctx,
                                          int max_inputs = 2) {
  if (ctx == nullptr) return "[]";

  std::ostringstream oss;
  oss << "[";
  const int total = ctx->num_inputs();
  const int limit = std::min(total, max_inputs);
  for (int i = 0; i < limit; ++i) {
    if (i > 0) oss << ",";
    oss << ctx->input(i).shape().DebugString();
  }
  if (total > limit) {
    if (limit > 0) oss << ",";
    oss << "...";
  }
  oss << "]";
  return oss.str();
}

class KernelTimingStatsRegistry {
 public:
  void Update(const std::string& kernel_name, const std::string& input_shape,
              double total_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::string key = kernel_name + " " + input_shape;
    auto& entry = stats_[key];
    if (entry.count == 0) {
      entry.kernel_name = kernel_name;
      entry.input_shape = input_shape;
      entry.min_ms = total_ms;
      entry.max_ms = total_ms;
    } else {
      entry.min_ms = std::min(entry.min_ms, total_ms);
      entry.max_ms = std::max(entry.max_ms, total_ms);
    }
    entry.count += 1;
    entry.total_ms += total_ms;
  }

  void PrintSummary() {
    const auto& cfg = GetKernelTimingConfig();
    if (!cfg.enabled || !cfg.stats) return;

    std::vector<KernelTimingStats> entries;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stats_.empty()) return;
      entries.reserve(stats_.size());
      for (const auto& item : stats_) {
        entries.push_back(item.second);
      }
    }

    std::sort(entries.begin(), entries.end(),
              [](const KernelTimingStats& lhs, const KernelTimingStats& rhs) {
                return lhs.total_ms > rhs.total_ms;
              });

    std::fprintf(stderr,
                 "=================================================================================\n");
    std::fprintf(stderr, "MUSA Kernel Debug Statistics\n");
    std::fprintf(stderr,
                 "=================================================================================\n");
    std::fprintf(stderr,
                 "%-16s %-20s %-10s %-12s %-12s %-12s %-12s\n",
                 "Kernel Name", "Input Shape", "Count", "Total(ms)", "Avg(ms)",
                 "Min(ms)", "Max(ms)");
    std::fprintf(stderr,
                 "---------------------------------------------------------------------------------\n");

    for (const auto& entry : entries) {
      const double avg = entry.count > 0
                             ? (entry.total_ms / static_cast<double>(entry.count))
                             : 0.0;
      std::fprintf(stderr,
                   "%-16s %-20s %-10llu %-12.3f %-12.3f %-12.3f %-12.3f\n",
                   entry.kernel_name.c_str(), entry.input_shape.c_str(),
                   static_cast<unsigned long long>(entry.count), entry.total_ms,
                   avg, entry.min_ms, entry.max_ms);
    }
    std::fprintf(stderr,
                 "=================================================================================\n");
  }

 private:
  std::mutex mu_;
  std::unordered_map<std::string, KernelTimingStats> stats_;
};

inline KernelTimingStatsRegistry& GlobalKernelTimingStats() {
  static KernelTimingStatsRegistry registry;
  return registry;
}

class KernelTimingStatsPrinter {
 public:
  ~KernelTimingStatsPrinter() { GlobalKernelTimingStats().PrintSummary(); }
};

inline KernelTimingStatsPrinter& GlobalKernelTimingStatsPrinter() {
  static KernelTimingStatsPrinter printer;
  return printer;
}

class KernelTimingScope {
 public:
  KernelTimingScope(const std::string& kernel_name,
                    const std::string& input_shape,
                    musaStream_t stream,
                    KernelTimingLayout layout = KernelTimingLayout())
      : kernel_name_(kernel_name),
        input_shape_(input_shape),
        stream_(stream),
        layout_(std::move(layout)) {
    const auto& cfg = GetKernelTimingConfig();
    level_ = cfg.level;
    stats_enabled_ = cfg.stats;
    active_ = ShouldTraceKernelName(kernel_name_);
    if (!active_) return;

    host_start_ns_ = NowNs();
    InitTotalDeviceTimer();
  }

  ~KernelTimingScope() {
    if (!active_) return;

    CloseOpenStagesWithWarning();

    const double host_total_ms = NsToMs(NowNs() - host_start_ns_);
    const double device_total_ms = FinalizeTotalDeviceTimer(host_total_ms);

    double stage_sum_ms = 0.0;
    for (const auto& item : stage_ms_) {
      stage_sum_ms += item.second;
    }

    if (stage_sum_ms > device_total_ms + 0.001) {
      AddWarning("Stage sum exceeds total. kernel=" + kernel_name_ +
                 ", stage_sum_ms=" + ToString(stage_sum_ms) +
                 ", device_total_ms=" + ToString(device_total_ms) +
                 ". This can happen with overlapping stages.");
    }

    const double other_ms = std::max(0.0, device_total_ms - stage_sum_ms);

    if (level_ >= 1) {
      std::lock_guard<std::mutex> lock(GetPrintMutex());
      PrintDeviceInfoOnceLocked();
      PrintWarningsLocked();

      if (level_ == 1) {
        PrintTotalOnlyLocked(host_total_ms, device_total_ms);
      } else {
        PrintDetailedLocked(host_total_ms, device_total_ms, other_ms);
      }
    }

    if (stats_enabled_) {
      GlobalKernelTimingStats().Update(kernel_name_, input_shape_,
                                       device_total_ms);
      (void)GlobalKernelTimingStatsPrinter();
    }
  }

  void TraceStart(const char* stage_name) {
    if (!active_) return;

    const std::string stage_id = MakeStageId(stage_name);
    if (stage_id.empty()) {
      WarnInvalidStage("START", stage_name);
      return;
    }

    if (stage_start_events_.find(stage_id) != stage_start_events_.end()) {
      AddWarning("Duplicate START without END. kernel=" + kernel_name_ +
                 ", stage=" + stage_id);
      return;
    }

    if (stage_ms_.find(stage_id) == stage_ms_.end()) {
      stage_order_.push_back(stage_id);
    }

    musaEvent_t start_event = nullptr;
    if (!CreateEvent(&start_event, "stage_start")) return;
    if (!RecordEvent(start_event, "stage_start")) {
      DestroyEvent(start_event);
      return;
    }
    stage_start_events_[stage_id] = start_event;
  }

  void TraceEnd(const char* stage_name) {
    if (!active_) return;

    const std::string stage_id = MakeStageId(stage_name);
    if (stage_id.empty()) {
      WarnInvalidStage("END", stage_name);
      return;
    }

    auto it = stage_start_events_.find(stage_id);
    if (it == stage_start_events_.end()) {
      AddWarning("END without matching START. kernel=" + kernel_name_ +
                 ", stage=" + stage_id);
      return;
    }

    musaEvent_t end_event = nullptr;
    if (!CreateEvent(&end_event, "stage_end")) {
      DestroyEvent(it->second);
      stage_start_events_.erase(it);
      return;
    }

    if (!RecordEvent(end_event, "stage_end") ||
        !SyncEvent(end_event, "stage_end")) {
      DestroyEvent(it->second);
      DestroyEvent(end_event);
      stage_start_events_.erase(it);
      return;
    }

    float elapsed_ms = 0.0f;
    if (ElapsedMs(it->second, end_event, &elapsed_ms, stage_id)) {
      stage_ms_[stage_id] += static_cast<double>(elapsed_ms);
    }

    DestroyEvent(it->second);
    DestroyEvent(end_event);
    stage_start_events_.erase(it);
  }

  // Backward-compatible one-shot API.
  void Trace(const char* stage_name) {
    TraceStart(stage_name);
    TraceEnd(stage_name);
  }

 private:
  static uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  static double NsToMs(uint64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
  }

  static std::string ToString(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return std::string(buf);
  }

  static std::mutex& GetPrintMutex() {
    static std::mutex print_mu;
    return print_mu;
  }

  static std::string GetDeviceInfo() {
    int count = 0;
    musaError_t count_status = musaGetDeviceCount(&count);

    int device = -1;
    musaError_t device_status = musaGetDevice(&device);

    std::ostringstream oss;
    oss << "device_id=" << device << ", device_count=" << count;

    if (device >= 0) {
      musaDeviceProp prop{};
      musaError_t prop_status = musaGetDeviceProperties(&prop, device);
      if (prop_status == musaSuccess) {
        oss << ", device_name=" << prop.name;
      } else {
        oss << ", prop_query_err=" << musaGetErrorString(prop_status);
      }
    }

    const char* visible_devices = std::getenv("MUSA_VISIBLE_DEVICES");
    if (visible_devices != nullptr && visible_devices[0] != '\0') {
      oss << ", MUSA_VISIBLE_DEVICES=" << visible_devices;
    }

    if (count_status != musaSuccess) {
      oss << ", count_query_err=" << musaGetErrorString(count_status);
    }
    if (device_status != musaSuccess) {
      oss << ", active_query_err=" << musaGetErrorString(device_status);
    }
    return oss.str();
  }

  musaStream_t GetTimingStream() const { return stream_; }

  bool CreateEvent(musaEvent_t* event, const char* tag) {
    musaError_t err = musaEventCreate(event);
    if (err != musaSuccess) {
      AddWarning(std::string("musaEventCreate failed on ") + tag +
                 ", err=" + musaGetErrorString(err));
      return false;
    }
    return true;
  }

  bool RecordEvent(musaEvent_t event, const char* tag) {
    musaError_t err = musaEventRecord(event, GetTimingStream());
    if (err != musaSuccess) {
      AddWarning(std::string("musaEventRecord failed on ") + tag +
                 ", err=" + musaGetErrorString(err));
      return false;
    }
    return true;
  }

  bool SyncEvent(musaEvent_t event, const char* tag) {
    musaError_t err = musaEventSynchronize(event);
    if (err != musaSuccess) {
      AddWarning(std::string("musaEventSynchronize failed on ") + tag +
                 ", err=" + musaGetErrorString(err));
      return false;
    }
    return true;
  }

  bool ElapsedMs(musaEvent_t start, musaEvent_t end, float* ms,
                 const std::string& stage_id) {
    musaError_t err = musaEventElapsedTime(ms, start, end);
    if (err != musaSuccess) {
      AddWarning("musaEventElapsedTime failed. kernel=" + kernel_name_ +
                 ", stage=" + stage_id +
                 ", err=" + musaGetErrorString(err));
      return false;
    }
    return true;
  }

  void DestroyEvent(musaEvent_t event) {
    if (event == nullptr) return;
    musaError_t err = musaEventDestroy(event);
    if (err != musaSuccess) {
      AddWarning(std::string("musaEventDestroy failed, err=") +
                 musaGetErrorString(err));
    }
  }

  void InitTotalDeviceTimer() {
    if (!CreateEvent(&total_start_event_, "total_start")) return;
    if (!RecordEvent(total_start_event_, "total_start")) {
      DestroyEvent(total_start_event_);
      total_start_event_ = nullptr;
      return;
    }
    total_start_event_valid_ = true;
  }

  double FinalizeTotalDeviceTimer(double fallback_ms) {
    if (!total_start_event_valid_) return fallback_ms;

    musaEvent_t total_end_event = nullptr;
    if (!CreateEvent(&total_end_event, "total_end")) {
      DestroyEvent(total_start_event_);
      total_start_event_ = nullptr;
      total_start_event_valid_ = false;
      return fallback_ms;
    }

    if (!RecordEvent(total_end_event, "total_end") ||
        !SyncEvent(total_end_event, "total_end")) {
      DestroyEvent(total_start_event_);
      DestroyEvent(total_end_event);
      total_start_event_ = nullptr;
      total_start_event_valid_ = false;
      return fallback_ms;
    }

    float elapsed_ms = 0.0f;
    double total_device_ms = fallback_ms;
    if (ElapsedMs(total_start_event_, total_end_event, &elapsed_ms, "TOTAL")) {
      total_device_ms = static_cast<double>(elapsed_ms);
    }

    DestroyEvent(total_start_event_);
    DestroyEvent(total_end_event);
    total_start_event_ = nullptr;
    total_start_event_valid_ = false;
    return total_device_ms;
  }

  static void PrintDeviceInfoOnceLocked() {
    static bool printed = false;
    if (printed) return;
    printed = true;
    std::fprintf(stderr, "\n[MUSA_KERNEL_TIMING_DEVICE] %s\n",
                 GetDeviceInfo().c_str());
  }

  static std::string MakeStageId(const char* stage_name) {
    if (stage_name == nullptr) return "";
    std::string id(stage_name);

    // Trim leading/trailing spaces to avoid accidental duplicates.
    const auto begin = id.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) return "";
    const auto end = id.find_last_not_of(" \t\n\r");
    return id.substr(begin, end - begin + 1);
  }

  void WarnInvalidStage(const char* action, const char* raw_name) {
    AddWarning(std::string("Invalid stage id on ") + action + ". kernel=" +
               kernel_name_ + ", raw=" +
               (raw_name == nullptr ? "<null>" : raw_name));
  }

  void ReportUnmatchedStarts() {
    for (const auto& item : stage_start_events_) {
      AddWarning("Unmatched START without END. kernel=" + kernel_name_ +
                 ", stage=" + item.first);
    }
  }

  void CloseOpenStagesWithWarning() {
    ReportUnmatchedStarts();
    for (auto& item : stage_start_events_) {
      DestroyEvent(item.second);
    }
    stage_start_events_.clear();
  }

  void AddWarning(const std::string& warning) {
    warnings_.push_back(warning);
  }

  void PrintWarningsLocked() {
    for (const auto& warning : warnings_) {
      std::fprintf(stderr, "[MUSA_KERNEL_TIMING_WARNING] %s\n", warning.c_str());
    }
  }

  void PrintOneStage(const std::string& name, double stage_ms) {
    std::fprintf(stderr, ", %s=%.3f", name.c_str(), stage_ms);
  }

  void PrintTotalOnlyLocked(double host_total_ms, double device_total_ms) {
    std::fprintf(stderr,
                 "[MUSA_KERNEL_TIMING] %s %s, host_total_ms=%.3f, "
                 "device_total_ms=%.3f\n",
                 kernel_name_.c_str(), input_shape_.c_str(), host_total_ms,
                 device_total_ms);
  }

  void PrintDetailedLocked(double host_total_ms, double device_total_ms,
                           double other_ms) {
    constexpr double kPrintEpsMs = 0.0005;

    std::fprintf(stderr,
                 "[MUSA_KERNEL_TIMING] %s %s, host_total_ms=%.3f, "
                 "device_total_ms=%.3f",
                 kernel_name_.c_str(), input_shape_.c_str(), host_total_ms,
                 device_total_ms);

    std::unordered_set<std::string> printed_ids;

    if (!layout_.empty()) {
      for (const auto& stage : layout_) {
        if (stage.stage_id.empty()) continue;
        const auto it = stage_ms_.find(stage.stage_id);
        const double ms = (it == stage_ms_.end()) ? 0.0 : it->second;
        if (!stage.show_zero && ms <= kPrintEpsMs) continue;

        const std::string display =
            stage.display_name.empty() ? stage.stage_id : stage.display_name;
        PrintOneStage(display, ms);
        printed_ids.insert(stage.stage_id);
      }
    }

    for (const auto& stage_id : stage_order_) {
      if (!layout_.empty() && printed_ids.find(stage_id) != printed_ids.end()) {
        continue;
      }
      const auto it = stage_ms_.find(stage_id);
      if (it == stage_ms_.end() || it->second <= kPrintEpsMs) continue;
      PrintOneStage(stage_id, it->second);
    }

    if (other_ms > kPrintEpsMs) {
      PrintOneStage("Other", other_ms);
    }

    std::fprintf(stderr, "\n");
  }

  bool active_ = false;
  int level_ = 0;
  bool stats_enabled_ = false;

  std::string kernel_name_;
  std::string input_shape_;
  musaStream_t stream_ = nullptr;

  uint64_t host_start_ns_ = 0;
  musaEvent_t total_start_event_ = nullptr;
  bool total_start_event_valid_ = false;
  std::unordered_map<std::string, musaEvent_t> stage_start_events_;
  std::unordered_map<std::string, double> stage_ms_;
  std::vector<std::string> stage_order_;
  KernelTimingLayout layout_;
  std::vector<std::string> warnings_;
};

}  // namespace timing
}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_TIMING_STAGE(stage_id, display_name, show_zero) \
  ::tensorflow::musa::timing::KernelTimingStageSpec((stage_id), (display_name), \
                                                    (show_zero))

#define MUSA_KERNEL_TIMING_LAYOUT(...) \
  ::tensorflow::musa::timing::KernelTimingLayout{__VA_ARGS__}

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME_AND_LAYOUT(ctx, kernel_name, layout) \
  ::tensorflow::musa::timing::KernelTimingScope __musa_kernel_timing_scope(      \
      (kernel_name), ::tensorflow::musa::timing::BuildInputShapeSummary(ctx),     \
      ::tensorflow::musa::GetMusaStreamByCtx((ctx)), (layout))

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, kernel_name)                     \
  ::tensorflow::musa::timing::KernelTimingScope __musa_kernel_timing_scope(      \
      (kernel_name), ::tensorflow::musa::timing::BuildInputShapeSummary(ctx),    \
      ::tensorflow::musa::GetMusaStreamByCtx((ctx)))

#define MUSA_KERNEL_TIMING_GUARD_WITH_LAYOUT(ctx, layout) \
  MUSA_KERNEL_TIMING_GUARD_WITH_NAME_AND_LAYOUT((ctx), (this)->def().op(), \
                                                (layout))

#define MUSA_KERNEL_TIMING_GUARD(ctx) \
  MUSA_KERNEL_TIMING_GUARD_WITH_NAME((ctx), (this)->def().op())

#define MUSA_KERNEL_TRACE_START(stage_name) \
  __musa_kernel_timing_scope.TraceStart((stage_name))

#define MUSA_KERNEL_TRACE_END(stage_name) \
  __musa_kernel_timing_scope.TraceEnd((stage_name))

#define MUSA_KERNEL_TRACE(stage_name) \
  __musa_kernel_timing_scope.Trace((stage_name))

#else

namespace tensorflow {
namespace musa {
namespace timing {

struct KernelTimingStageSpec {
  KernelTimingStageSpec(const std::string& id, const std::string& name,
                        bool show_zero_ms = false) {}
};

using KernelTimingLayout = std::vector<KernelTimingStageSpec>;

}  // namespace timing
}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_TIMING_STAGE(stage_id, display_name, show_zero) \
  ::tensorflow::musa::timing::KernelTimingStageSpec((stage_id), (display_name), \
                                                    (show_zero))

#define MUSA_KERNEL_TIMING_LAYOUT(...) \
  ::tensorflow::musa::timing::KernelTimingLayout{__VA_ARGS__}

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME_AND_LAYOUT(ctx, kernel_name, layout) \
  do {                                                                            \
  } while (false)

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, kernel_name) \
  do {                                                        \
  } while (false)

#define MUSA_KERNEL_TIMING_GUARD_WITH_LAYOUT(ctx, layout) \
  do {                                                     \
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

#endif  // MUSA_KERNEL_DEBUG

#endif  // MUSA_PLUGIN_SRC_UTILS_LOGGING_H_
