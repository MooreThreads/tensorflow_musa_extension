#include <musa_runtime.h>
#include <stdio.h>

#include <vector>

#include "device/musa_device.h"
#include "mu/device/musa_telemetry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
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
      return errors::Internal("Failed to get MUSA device count");
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

      // Use environment variable or default (0.9 of total memory)
      const char* mem_fraction_env = getenv("MUSA_MEMORY_FRACTION");
      double mem_fraction = 0.9;
      if (mem_fraction_env) {
        mem_fraction = std::stod(mem_fraction_env);
        if (mem_fraction < 0.1 || mem_fraction > 0.95) {
          LOG(WARNING)
              << "MUSA_MEMORY_FRACTION out of range (0.1-0.95), using 0.9";
          mem_fraction = 0.9;
        }
      }
      size_t memory_limit = static_cast<size_t>(total_memory * mem_fraction);

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

REGISTER_LOCAL_DEVICE_FACTORY("MUSA", MusaDeviceFactory, 210);

}  // namespace musa
}  // namespace tensorflow

extern "C" {
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
