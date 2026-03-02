#ifndef TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_
#define TENSORFLOW_MUSA_MU_KERNEL_REGISTER_H_

// ============================================================================
// CMake Build Type Auto-Detection
// ============================================================================
// Debug mode is automatically enabled when:
//   1. CMAKE_BUILD_TYPE=Debug (defines DEBUG or _DEBUG), or
//   2. MUSA_KERNEL_DEBUG is explicitly defined
//
// To disable debug even in Debug build:
//   #define MUSA_KERNEL_DEBUG_DISABLED before including this header
//
#if (defined(DEBUG) || defined(_DEBUG) || defined(MUSA_KERNEL_DEBUG)) && \
    !defined(MUSA_KERNEL_DEBUG_DISABLED)
#ifndef MUSA_KERNEL_DEBUG
#define MUSA_KERNEL_DEBUG 1
#endif
#endif

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../kernels/utils_op.h"
#include "./device_register.h"
#include "tensorflow/core/framework/op_kernel.h"

// Debug output macro - uses fprintf for reliability
#define MUSA_KERNEL_LOG(msg) do { \
  fprintf(stderr, "[MUSA_KERNEL] %s\n", msg); \
  fflush(stderr); \
} while(0)

// ============================================================================
// MUSA Kernel Profiling and Debug Macros
// ============================================================================
// 
// Usage:
//   1. Define MUSA_KERNEL_DEBUG before including this header to enable debug
//   2. Use MUSA_KERNEL_TRACE(ctx) at the beginning of Compute()
//   3. Or use manual timing with MUSA_KERNEL_TIMER_START/STOP
//
// Example:
//   void Compute(OpKernelContext* ctx) override {
//     MUSA_KERNEL_TRACE(ctx);  // Automatic timing and logging
//     // ... kernel implementation
//   }
//
// Or manual control:
//   void Compute(OpKernelContext* ctx) override {
//     MUSA_KERNEL_TIMER_START(op_name);
//     // ... kernel implementation
//     MUSA_KERNEL_TIMER_STOP(op_name);
//   }
//
// Environment variables:
//   MUSA_KERNEL_DEBUG=1        - Enable basic timing log
//   MUSA_KERNEL_DEBUG=2        - Enable detailed timing with args
//   MUSA_KERNEL_DEBUG_STATS=1  - Enable statistics aggregation
// ============================================================================

#ifdef MUSA_PROFILE
static std::mutex op_lock_;
#define MUSA_PROFILE_OP                                              \
  std::lock_guard<std::mutex> guard(op_lock_);                       \
  std::string kernel_label = "Musa" + OpKernel::def().op();          \
  std::string tf_op = OpKernel::name() + ":" + OpKernel::def().op(); \
  tensorflow::profiler::musa::AnnotatedTraceMe activity(             \
      [&] {                                                          \
        std::string op = tensorflow::profiler::TraceMeOp(            \
            absl::string_view(OpKernel::name()),                     \
            absl::string_view(kernel_label));                        \
        return tensorflow::profiler::TraceMeEncode(                  \
            kernel_label, {{"tf_op", tf_op},                         \
                           {"group_id", 0},                          \
                           {"is_eager", 0},                          \
                           {"context_id", "$$1"},                    \
                           {"correlation_id", correlation_id++},     \
                           {"kernel_details", "kernel_details"}});   \
      },                                                             \
      3);
#else
#define MUSA_PROFILE_OP
#endif

namespace tensorflow {
namespace musa {

// Forward declaration
class MusaDevice;

// ============================================================================
// Helper Functions (defined early for use in macros)
// ============================================================================

inline int GetDebugLevel() {
  static int level = -1;
  if (level < 0) {
    const char* env = std::getenv("MUSA_KERNEL_DEBUG");
    if (env) {
      level = std::atoi(env);
      fprintf(stderr, "[MUSA_KERNEL] Debug level set to %d (from MUSA_KERNEL_DEBUG=%s)\n", 
              level, env);
      fflush(stderr);
    } else {
      level = 0;
    }
  }
  return level;
}

inline bool GetDebugStats() {
  static int stats = -1;
  if (stats < 0) {
    const char* env = std::getenv("MUSA_KERNEL_DEBUG_STATS");
    stats = env ? std::atoi(env) : 0;
    if (stats > 0) {
      fprintf(stderr, "[MUSA_KERNEL] Statistics aggregation enabled\n");
      fflush(stderr);
    }
  }
  return stats > 0;
}

inline std::string GetInputShapes(OpKernelContext* ctx);

// ============================================================================
// Kernel Debug Timer Class
// ============================================================================
#ifdef MUSA_KERNEL_DEBUG

struct KernelTimingStats {
  int64 count = 0;
  double total_time_ms = 0.0;
  double min_time_ms = std::numeric_limits<double>::max();
  double max_time_ms = 0.0;
  
  void Record(double time_ms) {
    count++;
    total_time_ms += time_ms;
    min_time_ms = std::min(min_time_ms, time_ms);
    max_time_ms = std::max(max_time_ms, time_ms);
  }
  
  double AvgTime() const {
    return count > 0 ? total_time_ms / count : 0.0;
  }
};

// Global statistics storage
class KernelDebugStats {
 public:
  static KernelDebugStats& Instance() {
    static KernelDebugStats instance;
    return instance;
  }
  
  void RecordTiming(const std::string& op_name, double time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_[op_name].Record(time_ms);
  }
  
  void PrintStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stats_.empty()) return;
    
    fprintf(stderr, "\n====================================================================================================\n");
    fprintf(stderr, "MUSA Kernel Debug Statistics\n");
    fprintf(stderr, "====================================================================================================\n");
    fprintf(stderr, "%-40s %-12s %-12s %-12s %-12s %-12s\n", 
            "Kernel Name", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
    fprintf(stderr, "----------------------------------------------------------------------------------------------------\n");
    
    std::vector<std::pair<std::string, KernelTimingStats>> sorted_stats(
        stats_.begin(), stats_.end());
    std::sort(sorted_stats.begin(), sorted_stats.end(),
              [](const auto& a, const auto& b) {
                return a.second.total_time_ms > b.second.total_time_ms;
              });
    
    for (const auto& kv : sorted_stats) {
      const std::string& name = kv.first;
      const KernelTimingStats& stat = kv.second;
      fprintf(stderr, "%-40s %-12lld %-12.3f %-12.3f %-12.3f %-12.3f\n",
              name.c_str(),
              (long long)stat.count,
              stat.total_time_ms,
              stat.AvgTime(),
              (stat.min_time_ms == std::numeric_limits<double>::max() ? 0.0 : stat.min_time_ms),
              stat.max_time_ms);
    }
    fprintf(stderr, "====================================================================================================\n");
    fflush(stderr);
  }
  
  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
  }
  
 private:
  KernelDebugStats() = default;
  ~KernelDebugStats() {
    PrintStats();
  }
  
  std::mutex mutex_;
  std::unordered_map<std::string, KernelTimingStats> stats_;
};

// RAII Timer for automatic timing
class KernelTimer {
 public:
  explicit KernelTimer(const std::string& op_name, OpKernelContext* ctx = nullptr)
      : op_name_(op_name), ctx_(ctx), start_time_(std::chrono::high_resolution_clock::now()) {
    // Synchronize device before timing if in detailed mode
    if (GetDebugLevel() >= 2 && ctx_) {
      auto* device = static_cast<MusaDevice*>(ctx_->device());
      if (device) {
        musaStreamSynchronize(device->GetStream());
      }
    }
  }
  
  ~KernelTimer() {
    // Synchronize device before getting end time
    if (ctx_) {
      auto* device = static_cast<MusaDevice*>(ctx_->device());
      if (device) {
        musaStreamSynchronize(device->GetStream());
      }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    double time_ms = duration.count() / 1000.0;
    
    // Log individual execution
    if (GetDebugLevel() >= 1) {
      fprintf(stderr, "[MUSA_KERNEL] %s took %.3f ms\n", op_name_.c_str(), time_ms);
      fflush(stderr);
    }
    
    // Record statistics
    if (GetDebugStats()) {
      KernelDebugStats::Instance().RecordTiming(op_name_, time_ms);
    }
  }
  
 private:
  std::string op_name_;
  OpKernelContext* ctx_;
  std::chrono::high_resolution_clock::time_point start_time_;
};

// Helper function to get input shapes for detailed logging
inline std::string GetInputShapes(OpKernelContext* ctx) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    if (i > 0) ss << ",";
    const Tensor& input = ctx->input(i);
    ss << input.shape().DebugString();
  }
  ss << "]";
  return ss.str();
}

// ============================================================================
// Debug Macros
// ============================================================================

// Level 1: Basic timing log
#define MUSA_KERNEL_TRACE(ctx) \
  ::tensorflow::musa::KernelTimer _musa_kernel_timer(OpKernel::def().op(), (ctx))

// Level 2: Detailed timing with input shapes
#define MUSA_KERNEL_TRACE_DETAIL(ctx) \
  ::tensorflow::musa::KernelTimer _musa_kernel_timer( \
    OpKernel::def().op() + ::tensorflow::musa::GetInputShapes(ctx), (ctx))

// Manual timer control
#define MUSA_KERNEL_TIMER_START(name) \
  auto _musa_timer_start_##name = std::chrono::high_resolution_clock::now(); \
  do { \
    auto* _device = static_cast<::tensorflow::musa::MusaDevice*>(ctx->device()); \
    if (_device) musaStreamSynchronize(_device->GetStream()); \
  } while(0)

#define MUSA_KERNEL_TIMER_STOP(name) \
  do { \
    auto* _device = static_cast<::tensorflow::musa::MusaDevice*>(ctx->device()); \
    if (_device) musaStreamSynchronize(_device->GetStream()); \
    auto _musa_timer_end = std::chrono::high_resolution_clock::now(); \
    auto _musa_timer_duration = std::chrono::duration_cast<std::chrono::microseconds>( \
        _musa_timer_end - _musa_timer_start_##name); \
    double _musa_time_ms = _musa_timer_duration.count() / 1000.0; \
    fprintf(stderr, "[MUSA_KERNEL] %s took %.3f ms\n", #name, _musa_time_ms); \
    fflush(stderr); \
  } while(0)

// Print current statistics
#define MUSA_KERNEL_PRINT_STATS() \
  ::tensorflow::musa::KernelDebugStats::Instance().PrintStats()

// Reset statistics
#define MUSA_KERNEL_RESET_STATS() \
  ::tensorflow::musa::KernelDebugStats::Instance().Reset()

#else  // MUSA_KERNEL_DEBUG not defined

// No-op versions when debug is disabled
#define MUSA_KERNEL_TRACE(ctx) ((void)0)
#define MUSA_KERNEL_TRACE_DETAIL(ctx) ((void)0)
#define MUSA_KERNEL_TIMER_START(name) ((void)0)
#define MUSA_KERNEL_TIMER_STOP(name) ((void)0)
#define MUSA_KERNEL_PRINT_STATS() ((void)0)
#define MUSA_KERNEL_RESET_STATS() ((void)0)

inline std::string GetInputShapes(OpKernelContext* ctx) { return ""; }

#endif  // MUSA_KERNEL_DEBUG

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
