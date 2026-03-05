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
      std::fprintf(stderr, "%-16s %-20s %-10llu %-12.3f %-12.3f %-12.3f %-12.3f\n",
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
  KernelTimingScope(const std::string& kernel_name, const std::string& input_shape)
      : kernel_name_(kernel_name), input_shape_(input_shape) {
    const auto& cfg = GetKernelTimingConfig();
    level_ = cfg.level;
    stats_enabled_ = cfg.stats;
    active_ = ShouldTraceKernelName(kernel_name_);
    if (!active_) return;

    start_ns_ = NowNs();
  }

  ~KernelTimingScope() {
    if (!active_) return;

    const uint64_t end_ns = NowNs();
    CloseUnfinishedStages(end_ns);

    const double total_ms = NsToMs(end_ns - start_ns_);
    const double stage_sum_ms =
        mem_alloc_ms_ + mem_cpy_ms_ + kernel_ms_ + sync_ms_ + other_ms_;
    const double untraced_ms = std::max(0.0, total_ms - stage_sum_ms);
    const double other_total_ms = other_ms_ + untraced_ms;

    if (level_ == 1) {
      std::lock_guard<std::mutex> lock(GetPrintMutex());
      std::fprintf(stderr, "[MUSA_KERNEL_TIMING] %s %s | Total(ms) %.3f |\n",
                   kernel_name_.c_str(), input_shape_.c_str(), total_ms);
    } else if (level_ >= 2) {
      std::lock_guard<std::mutex> lock(GetPrintMutex());
      if (other_total_ms > 0.0005) {
        std::fprintf(stderr,
                     "[MUSA_KERNEL_TIMING] %s %s | Total(ms) %.3f | Mem Alloc "
                     "%.3f | Mem Cpy %.3f | Kernel %.3f | Sync %.3f | Other "
                     "%.3f |\n",
                     kernel_name_.c_str(), input_shape_.c_str(), total_ms,
                     mem_alloc_ms_, mem_cpy_ms_, kernel_ms_, sync_ms_,
                     other_total_ms);
      } else {
        std::fprintf(stderr,
                     "[MUSA_KERNEL_TIMING] %s %s | Total(ms) %.3f | Mem Alloc "
                     "%.3f | Mem Cpy %.3f | Kernel %.3f | Sync %.3f |\n",
                     kernel_name_.c_str(), input_shape_.c_str(), total_ms,
                     mem_alloc_ms_, mem_cpy_ms_, kernel_ms_, sync_ms_);
      }
    }

    if (stats_enabled_) {
      GlobalKernelTimingStats().Update(kernel_name_, input_shape_, total_ms);
      (void)GlobalKernelTimingStatsPrinter();
    }
  }

  void TraceStart(const char* stage_name) {
    if (!active_ || stage_name == nullptr) return;
    stage_start_ns_[NormalizeStage(stage_name)] = NowNs();
  }

  void TraceEnd(const char* stage_name) {
    if (!active_ || stage_name == nullptr) return;

    const std::string normalized = NormalizeStage(stage_name);
    auto it = stage_start_ns_.find(normalized);
    if (it == stage_start_ns_.end()) return;

    const uint64_t end_ns = NowNs();
    if (end_ns > it->second) {
      AccumulateStage(normalized, NsToMs(end_ns - it->second));
    }
    stage_start_ns_.erase(it);
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

  static bool Contains(const std::string& s, const char* token) {
    return s.find(token) != std::string::npos;
  }

  static std::string NormalizeStage(const char* stage_name) {
    std::string s = stage_name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return s;
  }

  static std::mutex& GetPrintMutex() {
    static std::mutex print_mu;
    return print_mu;
  }

  void AccumulateStage(const std::string& normalized, double delta_ms) {
    if (delta_ms <= 0.0) return;

    if (Contains(normalized, "alloc")) {
      mem_alloc_ms_ += delta_ms;
    } else if (Contains(normalized, "cpy") || Contains(normalized, "copy") ||
               Contains(normalized, "h2d") || Contains(normalized, "d2h") ||
               Contains(normalized, "d2d")) {
      mem_cpy_ms_ += delta_ms;
    } else if (Contains(normalized, "kernel") || Contains(normalized, "run") ||
               Contains(normalized, "compute")) {
      kernel_ms_ += delta_ms;
    } else if (Contains(normalized, "sync")) {
      sync_ms_ += delta_ms;
    } else {
      other_ms_ += delta_ms;
    }
  }

  void CloseUnfinishedStages(uint64_t end_ns) {
    for (const auto& item : stage_start_ns_) {
      if (end_ns > item.second) {
        AccumulateStage(item.first, NsToMs(end_ns - item.second));
      }
    }
    stage_start_ns_.clear();
  }

  bool active_ = false;
  int level_ = 0;
  bool stats_enabled_ = false;

  std::string kernel_name_;
  std::string input_shape_;

  uint64_t start_ns_ = 0;
  std::unordered_map<std::string, uint64_t> stage_start_ns_;

  double mem_alloc_ms_ = 0.0;
  double mem_cpy_ms_ = 0.0;
  double kernel_ms_ = 0.0;
  double sync_ms_ = 0.0;
  double other_ms_ = 0.0;
};

}  // namespace timing
}  // namespace musa
}  // namespace tensorflow

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, kernel_name)                \
  ::tensorflow::musa::timing::KernelTimingScope __musa_kernel_timing_scope( \
      (kernel_name), ::tensorflow::musa::timing::BuildInputShapeSummary(ctx))

#define MUSA_KERNEL_TIMING_GUARD(ctx) \
  MUSA_KERNEL_TIMING_GUARD_WITH_NAME((ctx), (this)->def().op())

#define MUSA_KERNEL_TRACE_START(stage_name) \
  __musa_kernel_timing_scope.TraceStart((stage_name))

#define MUSA_KERNEL_TRACE_END(stage_name) \
  __musa_kernel_timing_scope.TraceEnd((stage_name))

#define MUSA_KERNEL_TRACE(stage_name) \
  __musa_kernel_timing_scope.Trace((stage_name))

#else

#define MUSA_KERNEL_TIMING_GUARD_WITH_NAME(ctx, kernel_name) \
  do {                                                        \
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
