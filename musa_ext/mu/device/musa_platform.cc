/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mu/device/musa_platform.h"

#include <musa_runtime.h>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mu/device/musa_executor.h"
#include "mu/device/musa_platform_id.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
namespace musa {

MusaPlatform::MusaPlatform() : name_("MUSA") {}

MusaPlatform::~MusaPlatform() {}

Platform::Id MusaPlatform::id() const { return kMusaPlatformId; }

int MusaPlatform::VisibleDeviceCount() const {
  int count = 0;
  musaError_t err = musaGetDeviceCount(&count);
  if (err != musaSuccess) {
    LOG(WARNING) << "musaGetDeviceCount failed: " << musaGetErrorString(err);
    return 0;
  }
  return count;
}

const std::string& MusaPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
MusaPlatform::DescriptionForDevice(int ordinal) const {
  return MusaExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> MusaPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*>
MusaPlatform::ExecutorForDeviceWithPluginConfig(int ordinal,
                                                const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> MusaPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
MusaPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<MusaExecutor>(config.plugin_config), config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for MUSA device ordinal %d: %s",
            config.ordinal, init_status.ToString()));
  }

  return std::move(executor);
}

void MusaPlatform::RegisterTraceListener(std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register MUSA trace listener";
}

void MusaPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister MUSA trace listener";
}

}  // namespace musa

static void InitializeMusaPlatform() {
  // Disabling leak checking, MultiPlatformManager does not destroy its
  // registered platforms.
  std::unique_ptr<musa::MusaPlatform> platform(new musa::MusaPlatform);
  port::Status status = MultiPlatformManager::RegisterPlatform(std::move(platform));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register MUSA platform: " << status;
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(musa_platform,
                            stream_executor::InitializeMusaPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(musa_platform, multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     musa_platform);
