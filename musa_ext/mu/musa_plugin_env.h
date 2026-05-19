#ifndef TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_
#define TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_

#include <cstdlib>
#include <cstring>

namespace tensorflow {
namespace musa {
namespace plugin_env {

inline bool EnvIsOne(const char* name) {
  const char* p = std::getenv(name);
  return p && std::strcmp(p, "1") == 0;
}

inline bool UseLegacyCppDevicePath() {
  return EnvIsOne("TENSORFLOW_MUSA_USE_LEGACY_DEVICE");
}

inline bool PluggableSePathEnabled() { return !UseLegacyCppDevicePath(); }

// When "1", `ListPhysicalDevices` and the C++
// `MusaDeviceFactory::CreateDevices` path return FailedPrecondition on
// musaGetDeviceCount failure. When unset, list/enumeration returns OK with an
// empty list and `SE_InitPlugin` path `get_device_count` reports 0 (CPU-only /
// no-driver CI). Both paths use the same env for consistency.
inline bool StrictPhysicalDeviceEnum() {
  return EnvIsOne("MUSA_STRICT_DEVICE_ENUM");
}

inline bool SyncSeH2D() { return EnvIsOne("MUSA_SE_SYNC_H2D"); }

inline bool SyncSeStreamDependency() {
  return EnvIsOne("MUSA_SE_SYNC_STREAM_DEPENDENCY");
}

}  // namespace plugin_env
}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_MUSA_PLUGIN_ENV_H_
