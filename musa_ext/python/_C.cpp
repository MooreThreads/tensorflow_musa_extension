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

#include <cstdint>

#include "mu/device/caching_allocator.h"
#include "mu/device/host_caching_allocator.h"

namespace {

using ::tensorflow::musa::DeviceAllocatorBackend;
using ::tensorflow::musa::DeviceAllocatorBackendName;
using ::tensorflow::musa::DeviceCachingAllocator;
using ::tensorflow::musa::DeviceCachingAllocatorStats;
using ::tensorflow::musa::GetDeviceAllocatorBackend;
using ::tensorflow::musa::HostCachingAllocator;
using ::tensorflow::musa::HostCachingAllocatorStats;

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
      !put_u64("merges", s.merges) || !put_u64("segments", s.segments)) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

PyObject* DeviceAllocatorBackendStr(PyObject* /*self*/, PyObject* /*args*/) {
  return PyUnicode_FromString(
      DeviceAllocatorBackendName(GetDeviceAllocatorBackend()));
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
