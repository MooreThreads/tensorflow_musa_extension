#include <musa_runtime.h>

#include "musa_plugin_env.h"
#include "musa_executor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/platform/default/initialize.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

const Platform::Id kMusaPlatformId = reinterpret_cast<Platform::Id>(0x70000001);

class MusaPlatform : public Platform {
 public:
  MusaPlatform() : name_("MUSA") {}
  ~MusaPlatform() override {}

  Platform::Id id() const override { return kMusaPlatformId; }
  const std::string& Name() const override { return name_; }

  int VisibleDeviceCount() const override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      if (!::tensorflow::musa::plugin_env::StrictPhysicalDeviceEnum()) {
        VLOG(1) << "musaGetDeviceCount failed in MusaPlatform::VisibleDeviceCount; "
                   "returning 0 (set MUSA_STRICT_DEVICE_ENUM=1 to treat as error): "
                << musaGetErrorString(err);
      }
      return 0;
    }
    return count;
  }

  port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override {
    internal::DeviceDescriptionBuilder builder;
    builder.set_name("MUSA Device");
    builder.set_platform_version("1.0");
    return builder.Build();
  }

  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override {
    StreamExecutorConfig config;
    config.ordinal = ordinal;
    config.device_options = DeviceOptions::Default();
    return GetExecutor(config);
  }

  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& plugin_config) override {
    StreamExecutorConfig config;
    config.ordinal = ordinal;
    config.plugin_config = plugin_config;
    config.device_options = DeviceOptions::Default();
    return GetExecutor(config);
  }

  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override {
    return executor_cache_.GetOrCreate(
        config, [&]() { return GetUncachedExecutor(config); });
  }

  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override {
  }
  void UnregisterTraceListener(TraceListener* listener) override {}

 private:
  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) {
    auto executor = std::make_unique<MusaExecutor>(config.plugin_config);

    auto init_status = executor->Init(config.ordinal, config.device_options);
    if (!init_status.ok()) {
      return port::Status(
          port::error::INTERNAL,
          "Failed to initialize MUSA executor: " + init_status.ToString());
    }

    return std::make_unique<StreamExecutor>(this, std::move(executor),
                                            config.ordinal);
  }

  std::string name_;
  ExecutorCache executor_cache_;
};

void InitializeMusaPlatform() {
  std::unique_ptr<Platform> platform(new MusaPlatform);
  TF_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace musa
}  // namespace stream_executor

// Use TensorFlow module initializer (ordered with other stream_executor
// init) instead of a raw global static ctor; still gate on env for SE-only
// Pluggable path.
REGISTER_MODULE_INITIALIZER(musa_cxx_se_platform, {
  if (::tensorflow::musa::plugin_env::PluggableSePathEnabled()) {
    return;
  }
  stream_executor::musa::InitializeMusaPlatform();
});
