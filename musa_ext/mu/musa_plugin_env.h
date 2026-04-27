#ifndef TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_
#define TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_

#include <cstdlib>
#include <cstring>

// StreamExecutor C API (SE_InitPlugin) path.
// When "1", `libmusa_plugin.so` should be used as the TensorFlow PluggableDevice
// plugin (tensorflow-plugins layout): SE_InitPlugin registers the device, and
// the legacy C++ `REGISTER_LOCAL_DEVICE_FACTORY` + C++ MUSA StreamExecutor
// platform registration are skipped to avoid duplicate "MUSA" registration.
// When unset or not "1", the default is the historical path: C++ static
// registration (tf.load_op_library / import tensorflow_musa). If TensorFlow
// still invokes SE_InitPlugin in the same process, it returns UNIMPLEMENTED
// (only expected when a single .so is used both ways in error).
namespace tensorflow {
namespace musa {
namespace plugin_env {

inline bool PluggableSePathEnabled() {
  const char* p = std::getenv("MUSA_ENABLE_SE_PLUGIN");
  return p && std::strcmp(p, "1") == 0;
}

// When "1", `ListPhysicalDevices` and the C++ `MusaDeviceFactory::CreateDevices` path
// return FailedPrecondition on musaGetDeviceCount failure. When unset, list/enumeration
// returns OK with an empty list and `SE_InitPlugin` path `get_device_count` reports 0
// (CPU-only / no-driver CI). Both paths use the same env for consistency.
inline bool StrictPhysicalDeviceEnum() {
  const char* p = std::getenv("MUSA_STRICT_DEVICE_ENUM");
  return p && std::strcmp(p, "1") == 0;
}

}  // namespace plugin_env
}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_
