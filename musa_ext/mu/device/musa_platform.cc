#include <musa_runtime.h>

#include "musa_executor.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

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
    if (musaGetDeviceCount(&count) != musaSuccess) return 0;
    return count;
  }

  tsl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override {
    internal::DeviceDescriptionBuilder builder;
    builder.set_name("MUSA Device");
    builder.set_platform_version("1.0");
    return builder.Build();
  }

  tsl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override {
    StreamExecutorConfig config;
    config.ordinal = ordinal;
    config.device_options = DeviceOptions::Default();
    return GetExecutor(config);
  }

  tsl::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override {
    return executor_cache_.GetOrCreate(
        config, [&]() { return GetUncachedExecutor(config); });
  }

 private:
  tsl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) {
    auto executor = std::make_unique<MusaExecutor>();

    auto init_status = executor->Init(config.ordinal, config.device_options);
    if (!init_status.ok()) {
      return tsl::errors::Internal("Failed to initialize MUSA executor: " +
                                   init_status.ToString());
    }

    return std::make_unique<StreamExecutor>(this, std::move(executor),
                                            config.ordinal);
  }

  std::string name_;
  ExecutorCache executor_cache_;
};

static void InitializeMusaPlatform() {
  std::unique_ptr<Platform> platform(new MusaPlatform);
  tsl::Status register_status =
      MultiPlatformManager::RegisterPlatform(std::move(platform));
  if (!register_status.ok()) {
    auto existing = MultiPlatformManager::PlatformWithName(
        "MUSA", /*initialize_platform=*/false);
    if (existing.ok()) {
      return;
    }
    TF_CHECK_OK(register_status);
  }
}

}  // namespace musa
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(musa_platform,
                            stream_executor::musa::InitializeMusaPlatform());
