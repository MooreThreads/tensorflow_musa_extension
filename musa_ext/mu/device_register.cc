#include <musa_runtime.h>
#include <stdio.h>

#include <mutex>
#include <vector>

#include "device/musa_device.h"
#include "musa_plugin_env.h"
#include "mu/device/musa_telemetry.h"
#include "tensorflow/core/framework/device_factory.h"
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

class MusaDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      if (musa::plugin_env::StrictPhysicalDeviceEnum()) {
        return errors::FailedPrecondition(
            strings::StrCat("musaGetDeviceCount failed: ", musaGetErrorString(err)));
      }
      VLOG(1) << "musaGetDeviceCount failed; returning empty physical device list "
                 "(set MUSA_STRICT_DEVICE_ENUM=1 to treat this as an error): "
              << musaGetErrorString(err);
      return Status::OK();
    }

    for (int i = 0; i < count; ++i) {
      devices->push_back(strings::StrCat("/physical_device:MUSA:", i));
    }
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      if (musa::plugin_env::StrictPhysicalDeviceEnum()) {
        return errors::FailedPrecondition(
            strings::StrCat("musaGetDeviceCount failed: ", musaGetErrorString(err)));
      }
      VLOG(1) << "musaGetDeviceCount failed; skipping MUSA device creation "
                 "(set MUSA_STRICT_DEVICE_ENUM=1 to treat this as an error): "
              << musaGetErrorString(err);
      return Status::OK();
    }

    auto platform_status =
        ::stream_executor::MultiPlatformManager::PlatformWithName("MUSA");
    if (!platform_status.ok()) {
      return platform_status.status();
    }
    auto* platform = platform_status.ValueOrDie();

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
          new MusaDevice(Env::Default(), attr, i, executor)));
    }
    return Status::OK();
  }
};

}  // namespace musa
}  // namespace tensorflow

extern "C" {
void __attribute__((constructor)) OnMusaPluginLoad() {
  // Single entry point: register C++ MUSA device factory unless SE-only
  // Pluggable path is selected (MUSA_ENABLE_SE_PLUGIN=1 before dlopen).
  static std::once_flag k_register_musa_device_factory;
  std::call_once(k_register_musa_device_factory, [] {
    if (tensorflow::musa::plugin_env::PluggableSePathEnabled()) {
      LOG(INFO) << "[MUSA] Skipping C++ DeviceFactory::Register(\"MUSA\") "
                   "because MUSA_ENABLE_SE_PLUGIN=1 (use SE_InitPlugin path). "
                   "Set this env before loading libmusa_plugin.so.";
      return;
    }
    tensorflow::DeviceFactory::Register(
        "MUSA", new tensorflow::musa::MusaDeviceFactory(), 210,
        /*is_pluggable_device*/ false);
    LOG(INFO) << "[MUSA] Registered C++ DeviceFactory for device type MUSA.";
  });

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
