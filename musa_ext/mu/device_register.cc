#include <musa_runtime.h>
#include <stdio.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "device/musa_device.h"
#include "mu/device/musa_telemetry.h"
#include "mu/runtime_config_c_api.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
void ForceMusaOptimizationPassRegistration();
}

namespace tensorflow {
namespace musa {
namespace {

constexpr bool kDefaultMusaAllowGrowth = false;

std::atomic<bool> g_musa_allow_growth(kDefaultMusaAllowGrowth);

bool ParseAllowGrowthEnv(const char* env_name, const char* env_value,
                         bool* allow_growth) {
  if (std::strcmp("false", env_value) == 0) {
    *allow_growth = false;
    return true;
  }

  if (std::strcmp("true", env_value) == 0) {
    *allow_growth = true;
    return true;
  }

  LOG(ERROR) << env_name << " is set but could not be parsed: \"" << env_value
             << "\". Valid values are \"true\" or \"false\". Ignoring it.";
  return false;
}

bool GetMusaAllowGrowthValue() {
  const char* force_allow_growth = std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  bool allow_growth = kDefaultMusaAllowGrowth;
  if (force_allow_growth != nullptr &&
      ParseAllowGrowthEnv("TF_FORCE_GPU_ALLOW_GROWTH", force_allow_growth,
                          &allow_growth)) {
    return allow_growth;
  }

  return g_musa_allow_growth.load();
}

}  // namespace

void SetMusaAllowGrowthOverride(bool enabled) {
  g_musa_allow_growth.store(enabled);
}

class MusaDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      return Status::OK();
    }

    for (int i = 0; i < count; ++i) {
      devices->push_back(strings::StrCat("/physical_device:MUSA:", i));
    }
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    (void)options;
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      return errors::Internal("Failed to get MUSA device count");
    }

    auto platform_status =
        ::stream_executor::MultiPlatformManager::PlatformWithName("MUSA");
    if (!platform_status.ok()) {
      return platform_status.status();
    }
    auto* platform = platform_status.ValueOrDie();
    const bool allow_growth = GetMusaAllowGrowthValue();

    for (int i = 0; i < count; ++i) {
      DeviceAttributes attr;
      string name = strings::StrCat(name_prefix, "/device:MUSA:", i);
      attr.set_name(name);
      attr.set_device_type("MUSA");

      // FIX: Dynamically get GPU memory and set correct memory_limit
      // to match BFCAllocator configuration in musa_device.cc
      musaSetDevice(i);
      size_t total_memory = 0, free_memory = 0;
      musaMemGetInfo(&free_memory, &total_memory);
      size_t memory_limit =
          static_cast<size_t>(free_memory * 0.9);  // 90% of free memory
      attr.set_memory_limit(memory_limit);

      attr.mutable_locality()->set_bus_id(i);
      attr.set_physical_device_desc(strings::StrCat("device: MUSA device ", i));

      auto executor_status = platform->ExecutorForDevice(i);
      if (!executor_status.ok()) {
        return executor_status.status();
      }
      auto* executor = executor_status.ValueOrDie();

      devices->push_back(std::unique_ptr<Device>(
          new MusaDevice(Env::Default(), attr, i, executor, allow_growth)));
    }
    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("MUSA", MusaDeviceFactory, 210);

}  // namespace musa
}  // namespace tensorflow

extern "C" {
void __attribute__((visibility("default"))) TFMusaSetAllowGrowth(int enabled) {
  ::tensorflow::musa::SetMusaAllowGrowthOverride(enabled != 0);
}

void __attribute__((constructor)) OnMusaPluginLoad() {
  // Initialize telemetry system from environment variables
  auto config = ::tensorflow::musa::TelemetryConfig::FromEnv();
  if (config.enabled) {
    ::tensorflow::musa::MusaTelemetry::Instance().Initialize(config);
    LOG(INFO) << "[MUSA] Telemetry system initialized. "
              << "Log path: "
              << (config.log_path.empty() ? "stderr" : config.log_path)
              << ", Buffer size: " << config.buffer_size;
  }
  // fprintf(stderr, "\n>>>> [MUSA] SUCCESS: MUSA Factory Object Registered via
  // Global Constructor! <<<<\n");
}

void __attribute__((destructor)) OnMusaPluginUnload() {
  // Shutdown telemetry system
  ::tensorflow::musa::MusaTelemetry::Instance().Shutdown();
}
}
// extern "C" void ForceLinkMusaAmpOptimizer();
