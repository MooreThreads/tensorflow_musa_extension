/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

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

// Scaffold for the `tensorflow_musa._C` Python extension module.
//
// Goal of commit C2 (plan §5.2.1) is *only* to prove the three-target build
// split works end-to-end: this extension links against libmusa_core.so and
// must dlopen cleanly when imported. The surface exposed here is
// deliberately minimal -- one build-identity probe and a thin wrapper over
// the host caching allocator stats that validates cross-library singleton
// sharing. Real memory / device / stream APIs land with commit C6.
//
// We use the bare CPython C API (no pybind11 yet) so the plugin does not
// take on an extra build-time dependency until C6, when we gain real
// binding surface that benefits from pybind11's type casters.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <musa_runtime.h>

#include <cstdint>
#include <vector>

#include "mu/device/allocator_config.h"
#include "mu/device/caching_allocator.h"
#include "mu/device/driver_api.h"
#include "mu/device/expandable_segment.h"
#include "mu/device/host_caching_allocator.h"

namespace {

using ::tensorflow::musa::AllocatorConfig;
using ::tensorflow::musa::DeviceAllocatorBackend;
using ::tensorflow::musa::DeviceAllocatorBackendName;
using ::tensorflow::musa::DeviceCachingAllocator;
using ::tensorflow::musa::DeviceCachingAllocatorStats;
using ::tensorflow::musa::DeviceSegmentInfo;
using ::tensorflow::musa::GetDeviceAllocatorBackend;
using ::tensorflow::musa::HostCachingAllocator;
using ::tensorflow::musa::HostCachingAllocatorStats;
using ::tensorflow::musa::IsVmmAvailable;
using ::tensorflow::musa::IsVmmSupportedForDevice;
using ::tensorflow::musa::QueryMinAllocationGranularity;

PyObject* BuildStatsDict(const HostCachingAllocatorStats& s) {
  PyObject* d = PyDict_New();
  if (d == nullptr) return nullptr;
  auto put_u64 = [&](const char* key, std::uint64_t v) -> bool {
    PyObject* val = PyLong_FromUnsignedLongLong(v);
    if (val == nullptr) return false;
    int rc = PyDict_SetItemString(d, key, val);
    Py_DECREF(val);
    return rc == 0;
  };
  if (!put_u64("in_use_bytes", s.in_use_bytes) ||
      !put_u64("cached_bytes", s.cached_bytes) ||
      !put_u64("total_reserved_bytes", s.total_reserved_bytes) ||
      !put_u64("peak_bytes", s.peak_bytes) ||
      !put_u64("alloc_requests", s.alloc_requests) ||
      !put_u64("cache_hits", s.cache_hits) ||
      !put_u64("cache_misses", s.cache_misses) ||
      !put_u64("pool_cap_rejections", s.pool_cap_rejections) ||
      !put_u64("record_stream_count", s.record_stream_count)) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

// Trivial build-identity probe used by tests to confirm the extension
// loaded and links against libmusa_core.so.
PyObject* IsLoaded(PyObject* /*self*/, PyObject* /*args*/) { Py_RETURN_TRUE; }

// Returns the current HostCachingAllocator stats as a dict. Reaching into
// this function validates the cross-library singleton: libmusa_plugin.so
// and tensorflow_musa._C.so must observe the *same* allocator state. A
// zero-filled result means the allocator has not been touched yet (valid
// when `tensorflow_musa._C` is imported before TF loads the plugin).
PyObject* HostAllocatorStats(PyObject* /*self*/, PyObject* /*args*/) {
  HostCachingAllocatorStats s = HostCachingAllocator::Instance().GetStats();
  return BuildStatsDict(s);
}

PyObject* DeviceAllocatorStats(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  DeviceCachingAllocatorStats s =
      DeviceCachingAllocator::For(ordinal).GetStats();
  PyObject* d = PyDict_New();
  if (d == nullptr) return nullptr;
  auto put_u64 = [&](const char* key, std::uint64_t v) -> bool {
    PyObject* val = PyLong_FromUnsignedLongLong(v);
    if (val == nullptr) return false;
    int rc = PyDict_SetItemString(d, key, val);
    Py_DECREF(val);
    return rc == 0;
  };
  if (!put_u64("in_use_bytes", s.in_use_bytes) ||
      !put_u64("reserved_bytes", s.reserved_bytes) ||
      !put_u64("cached_bytes", s.cached_bytes) ||
      !put_u64("peak_in_use_bytes", s.peak_in_use_bytes) ||
      !put_u64("alloc_requests", s.alloc_requests) ||
      !put_u64("cache_hits", s.cache_hits) ||
      !put_u64("cache_misses", s.cache_misses) ||
      !put_u64("oom_events", s.oom_events) || !put_u64("splits", s.splits) ||
      !put_u64("merges", s.merges) || !put_u64("segments", s.segments) ||
      !put_u64("limit_bytes", s.limit_bytes) ||
      !put_u64("total_device_bytes", s.total_device_bytes)) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

PyObject* DeviceSegmentSnapshot(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  std::vector<DeviceSegmentInfo> segs =
      DeviceCachingAllocator::For(ordinal).GetSegmentSnapshot();
  PyObject* list = PyList_New(0);
  if (list == nullptr) return nullptr;
  for (const auto& info : segs) {
    PyObject* d = PyDict_New();
    if (d == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    auto put_u64 = [&](const char* key, std::uint64_t v) -> bool {
      PyObject* val = PyLong_FromUnsignedLongLong(v);
      if (val == nullptr) return false;
      int rc = PyDict_SetItemString(d, key, val);
      Py_DECREF(val);
      return rc == 0;
    };
    auto put_i = [&](const char* key, long v) -> bool {
      PyObject* val = PyLong_FromLong(v);
      if (val == nullptr) return false;
      int rc = PyDict_SetItemString(d, key, val);
      Py_DECREF(val);
      return rc == 0;
    };
    auto put_bool = [&](const char* key, bool v) -> bool {
      PyObject* val = PyBool_FromLong(v ? 1 : 0);
      int rc = PyDict_SetItemString(d, key, val);
      Py_DECREF(val);
      return rc == 0;
    };
    if (!put_i("device", info.device) || !put_u64("address", info.address) ||
        !put_u64("size", info.size) || !put_u64("in_use", info.in_use) ||
        !put_i("num_blocks", info.num_blocks) ||
        !put_i("num_free_blocks", info.num_free_blocks) ||
        !put_u64("largest_free_block", info.largest_free_block) ||
        !put_bool("is_expandable", info.is_expandable)) {
      Py_DECREF(d);
      Py_DECREF(list);
      return nullptr;
    }
    if (PyList_Append(list, d) != 0) {
      Py_DECREF(d);
      Py_DECREF(list);
      return nullptr;
    }
    Py_DECREF(d);
  }
  return list;
}

PyObject* DeviceSetMemoryFraction(PyObject* /*self*/, PyObject* args) {
  double fraction = 0.0;
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "d|i", &fraction, &ordinal)) return nullptr;
  uint64_t limit =
      DeviceCachingAllocator::For(ordinal).SetMemoryFraction(fraction);
  return PyLong_FromUnsignedLongLong(limit);
}

PyObject* DeviceSetMemoryLimitBytes(PyObject* /*self*/, PyObject* args) {
  // Accept Python int (unsigned 64-bit); ordinal optional.
  unsigned long long bytes = 0;
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "K|i", &bytes, &ordinal)) return nullptr;
  DeviceCachingAllocator::For(ordinal).SetMemoryLimitBytes(
      static_cast<uint64_t>(bytes));
  Py_RETURN_NONE;
}

PyObject* DeviceResetPeakStats(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  DeviceCachingAllocator::For(ordinal).ResetPeakStats();
  Py_RETURN_NONE;
}

PyObject* DeviceLastOomMessage(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  std::string msg = DeviceCachingAllocator::For(ordinal).GetLastOomMessage();
  return PyUnicode_FromStringAndSize(msg.data(),
                                     static_cast<Py_ssize_t>(msg.size()));
}

PyObject* DeviceMemoryUsage(PyObject* /*self*/, PyObject* args) {
  // Returns a (free_bytes, total_bytes) tuple from musaMemGetInfo on
  // the given ordinal. Complements the caching allocator's stats by
  // exposing the driver's view of device memory.
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  musaError_t e = musaSetDevice(ordinal);
  if (e != musaSuccess) {
    (void)musaGetLastError();
    PyErr_Format(PyExc_RuntimeError, "musaSetDevice(%d) failed", ordinal);
    return nullptr;
  }
  size_t free_b = 0, total_b = 0;
  if (musaMemGetInfo(&free_b, &total_b) != musaSuccess) {
    (void)musaGetLastError();
    PyErr_SetString(PyExc_RuntimeError, "musaMemGetInfo failed");
    return nullptr;
  }
  return Py_BuildValue("(KK)", static_cast<unsigned long long>(free_b),
                       static_cast<unsigned long long>(total_b));
}

PyObject* DeviceAllocatorBackendStr(PyObject* /*self*/, PyObject* /*args*/) {
  return PyUnicode_FromString(
      DeviceAllocatorBackendName(GetDeviceAllocatorBackend()));
}

// VMM / expandable segments introspection. These are process-wide
// read-only probes — they never touch the allocator's state and never
// hold its lock, so they are safe to call from any thread at any time.

PyObject* VmmAvailable(PyObject* /*self*/, PyObject* /*args*/) {
  return PyBool_FromLong(IsVmmAvailable() ? 1 : 0);
}

PyObject* VmmSupported(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  return PyBool_FromLong(IsVmmSupportedForDevice(ordinal) ? 1 : 0);
}

// Returns the driver-reported minimum granularity (in bytes) for
// pinned device memory on `ordinal`, or 0 if the driver is unavailable
// or the query fails. Useful both for runbooks ("what sizes will
// expandable_segments round to?") and for tests.
PyObject* VmmGranularity(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  std::size_t g = QueryMinAllocationGranularity(ordinal);
  return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(g));
}

// Expose the parsed TF_MUSA_ALLOC_CONF as a dict. Keeps all fields
// present even when the env var is unset so consumers (tests, user
// scripts) can rely on the schema.
PyObject* AllocatorConfigDict(PyObject* /*self*/, PyObject* /*args*/) {
  const AllocatorConfig& c = AllocatorConfig::Instance();
  PyObject* d = PyDict_New();
  if (d == nullptr) return nullptr;
  auto put_bool = [&](const char* k, bool v) -> bool {
    PyObject* o = PyBool_FromLong(v ? 1 : 0);
    int rc = PyDict_SetItemString(d, k, o);
    Py_DECREF(o);
    return rc == 0;
  };
  auto put_u64 = [&](const char* k, std::uint64_t v) -> bool {
    PyObject* o = PyLong_FromUnsignedLongLong(v);
    if (o == nullptr) return false;
    int rc = PyDict_SetItemString(d, k, o);
    Py_DECREF(o);
    return rc == 0;
  };
  auto put_int = [&](const char* k, long v) -> bool {
    PyObject* o = PyLong_FromLong(v);
    if (o == nullptr) return false;
    int rc = PyDict_SetItemString(d, k, o);
    Py_DECREF(o);
    return rc == 0;
  };
  auto put_double = [&](const char* k, double v) -> bool {
    PyObject* o = PyFloat_FromDouble(v);
    if (o == nullptr) return false;
    int rc = PyDict_SetItemString(d, k, o);
    Py_DECREF(o);
    return rc == 0;
  };
  auto put_str = [&](const char* k, const std::string& v) -> bool {
    PyObject* o = PyUnicode_FromStringAndSize(
        v.data(), static_cast<Py_ssize_t>(v.size()));
    if (o == nullptr) return false;
    int rc = PyDict_SetItemString(d, k, o);
    Py_DECREF(o);
    return rc == 0;
  };
  if (!put_bool("expandable_segments", c.expandable_segments()) ||
      !put_u64("max_split_size_bytes", c.max_split_size_bytes()) ||
      !put_int("roundup_power2_divisions", c.roundup_power2_divisions()) ||
      !put_double("garbage_collection_threshold",
                  c.garbage_collection_threshold()) ||
      !put_str("raw", c.raw())) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

PyObject* DeviceEmptyCache(PyObject* /*self*/, PyObject* args) {
  int ordinal = 0;
  if (!PyArg_ParseTuple(args, "|i", &ordinal)) return nullptr;
  uint64_t released = DeviceCachingAllocator::For(ordinal).EmptyCache();
  return PyLong_FromUnsignedLongLong(released);
}

PyMethodDef kMethods[] = {
    {"_is_loaded", IsLoaded, METH_NOARGS,
     "Return True iff the native _C module loaded successfully."},
    {"_host_allocator_stats", HostAllocatorStats, METH_NOARGS,
     "Return a dict snapshot of HostCachingAllocator statistics."},
    {"_device_allocator_stats", DeviceAllocatorStats, METH_VARARGS,
     "Return a dict snapshot of the DeviceCachingAllocator for the given "
     "device ordinal (default 0)."},
    {"_device_allocator_backend", DeviceAllocatorBackendStr, METH_NOARGS,
     "Return the active device allocator backend name: 'caching' or "
     "'passthrough'."},
    {"_device_empty_cache", DeviceEmptyCache, METH_VARARGS,
     "Release every fully-free cached segment on the given ordinal and "
     "return the number of bytes returned to the driver."},
    {"_device_segment_snapshot", DeviceSegmentSnapshot, METH_VARARGS,
     "Return a list of dicts describing every live segment owned by the "
     "DeviceCachingAllocator on the given ordinal (default 0)."},
    {"_device_set_memory_fraction", DeviceSetMemoryFraction, METH_VARARGS,
     "Set a hard per-process memory cap as a fraction (0 < f <= 1) of the "
     "device's total memory. Returns the resulting byte limit (0 if the "
     "cap was cleared)."},
    {"_device_set_memory_limit_bytes", DeviceSetMemoryLimitBytes, METH_VARARGS,
     "Set a hard per-process memory cap in bytes (0 clears the cap)."},
    {"_device_reset_peak_stats", DeviceResetPeakStats, METH_VARARGS,
     "Reset the per-device peak_in_use_bytes counter to the current "
     "in_use_bytes value."},
    {"_device_last_oom_message", DeviceLastOomMessage, METH_VARARGS,
     "Return the most recent OOM diagnostic string emitted by the "
     "DeviceCachingAllocator (empty string if no OOM has occurred)."},
    {"_device_memory_usage", DeviceMemoryUsage, METH_VARARGS,
     "Return (free_bytes, total_bytes) reported by musaMemGetInfo for the "
     "given ordinal (default 0)."},
    {"_vmm_available", VmmAvailable, METH_NOARGS,
     "Return True iff libmusa.so exports the full VMM API "
     "(muMemAddressReserve / muMemCreate / muMemMap / muMemSetAccess / "
     "muMemGetAllocationGranularity)."},
    {"_vmm_supported", VmmSupported, METH_VARARGS,
     "Return True iff the given device ordinal advertises "
     "MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED. Implies "
     "_vmm_available()."},
    {"_vmm_granularity", VmmGranularity, METH_VARARGS,
     "Return the driver-reported minimum allocation granularity (in "
     "bytes) for pinned device memory on the given ordinal; 0 when "
     "unavailable."},
    {"_allocator_config", AllocatorConfigDict, METH_NOARGS,
     "Return a dict snapshot of the parsed TF_MUSA_ALLOC_CONF env "
     "variable (expandable_segments, max_split_size_bytes, ...)."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "tensorflow_musa._C",
    "Native helpers for the tensorflow_musa Python package.",
    /*m_size=*/-1,
    kMethods,
    /*m_slots=*/nullptr,
    /*m_traverse=*/nullptr,
    /*m_clear=*/nullptr,
    /*m_free=*/nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__C(void) { return PyModule_Create(&kModuleDef); }
