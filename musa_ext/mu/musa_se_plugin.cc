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

// StreamExecutor C API (PluggableDevice) implementation for MUSA. This file
// provides `SE_InitPlugin` and the `plugin_*` entry points declared in
// device_register.h, mapping to MUSART runtime while leaving existing
// C++-registered MusaDevice/allocator code paths intact.

#include <musa_runtime.h>

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "musa_plugin_env.h"
#include "mu/musa_plugin_sp_stream.h"
#include "mu/musa_runtime_adapter.h"
#include "mu/musa_runtime_registry.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/logging.h"

// TensorFlow headers only forward-declare these; the plugin must define the full
// struct types in the global namespace (same tag names) so SP_Stream/SP_Event/
// SP_Timer pointers are complete when dereferenced.
struct SP_Event_st {
  musaEvent_t event;
};

struct SP_Timer_st {
  musaEvent_t start_event;
  musaEvent_t end_event;
  bool has_stop;
};

// NAME_MTGPU storage (declared in device_register.h)
extern "C" const char NAME_MTGPU[] = "MUSA";

namespace {

static const char kPlatformName[] = "MUSA";
static const char kPlatformType[] = "MUSA";
static const char kVendorMthreads[] = "MooreThreads";

static musaStream_t StreamPtr(SP_Stream s) {
  if (!s || s->magic != kMusaPluginSpStreamMagic) return nullptr;
  return s->stream;
}

static void MallocOrBadAlloc(TF_Status* status) {
  TF_SetStatus(status, TF_RESOURCE_EXHAUSTED, "malloc failed");
}

static int Ordinal(const SP_Device* d) {
  if (!d) return -1;
  return ::tensorflow::musa::runtime::GetDeviceOrdinal(
      d->device_handle, d->ordinal);
}

// --- SP_StreamExecutor callback implementations (names match SP_* vtable) ---

void plugin_se_allocate(const SP_Device* device, uint64_t size, int64_t,
                        SP_DeviceMemoryBase* mem) {
  if (!mem) return;
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->ext = nullptr;
  mem->opaque = nullptr;
  mem->size = 0;
  mem->payload = 0;
  if (size == 0) return;
  const int ordinal = Ordinal(device);
  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err != musaSuccess) {
    LOG(ERROR) << "plugin_se_allocate: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
    // Preserve requested size so callers can distinguish from a real zero-size
    // allocation.
    mem->size = size;
    return;
  }
  void* p = nullptr;
  musaError_t err = musaMalloc(&p, size);
  if (err != musaSuccess) {
    return;
  }
  mem->opaque = p;
  mem->size = size;
}

void plugin_se_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  if (!mem || !mem->opaque) return;
  const int ordinal = Ordinal(device);
  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err == musaSuccess) {
    (void)musaFree(mem->opaque);
  } else {
    LOG(ERROR) << "plugin_se_deallocate: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
  }
  mem->opaque = nullptr;
  mem->size = 0;
}

void* plugin_se_host_memory_allocate(const SP_Device* device, uint64_t size) {
  if (size == 0) return nullptr;
  const int ordinal = Ordinal(device);
  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err != musaSuccess) {
    LOG(ERROR) << "plugin_se_host_memory_allocate: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
    return nullptr;
  }
  void* p = nullptr;
  musaError_t err = musaHostAlloc(&p, size, musaHostAllocDefault);
  if (err != musaSuccess) return nullptr;
  return p;
}

void plugin_se_host_memory_deallocate(const SP_Device* device, void* mem) {
  if (!mem) return;
  const int ordinal = Ordinal(device);
  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err == musaSuccess) {
    (void)musaFreeHost(mem);
  } else {
    LOG(ERROR) << "plugin_se_host_memory_deallocate: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
  }
}

void* plugin_se_unified_memory_allocate(const SP_Device*, uint64_t) {
  return nullptr;
}

void plugin_se_unified_memory_deallocate(const SP_Device*, void*) {}

TF_Bool plugin_se_get_allocator_stats(const SP_Device*, SP_AllocatorStats*) {
  return 0;
}

TF_Bool plugin_se_device_memory_usage(const SP_Device* device, int64_t* free_out,
                                            int64_t* total_out) {
  if (!free_out || !total_out) return 0;
  const int ordinal = Ordinal(device);
  musaError_t set_dev_err = musaSetDevice(ordinal);
  if (set_dev_err != musaSuccess) {
    LOG(ERROR) << "plugin_se_device_memory_usage: musaSetDevice(" << ordinal
               << ") failed: " << musaGetErrorString(set_dev_err);
    return 0;
  }
  size_t free_m = 0, total = 0;
  musaError_t err = musaMemGetInfo(&free_m, &total);
  if (err != musaSuccess) return 0;
  *free_out = static_cast<int64_t>(free_m);
  *total_out = static_cast<int64_t>(total);
  return 1;
}

void plugin_se_create_stream(const SP_Device* device, SP_Stream* stream,
                             TF_Status* status) {
  if (!stream) return;
  *stream = nullptr;
  auto* s = new (std::nothrow) SP_Stream_st();
  if (!s) {
    MallocOrBadAlloc(status);
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) {
    delete s;
    return;
  }
  musaError_t   err = musaStreamCreate(&s->stream);
  if (err != musaSuccess) {
    delete s;
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamCreate");
    return;
  }
  s->magic = kMusaPluginSpStreamMagic;
  *stream = s;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_se_destroy_stream(const SP_Device* device, SP_Stream stream) {
  if (!stream) return;
  TF_Status* st = TF_NewStatus();
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), st);
  if (TF_GetCode(st) == TF_OK) {
    (void)musaStreamDestroy(stream->stream);
  }
  TF_DeleteStatus(st);
  delete stream;
}

// musaStreamWaitEvent() queues the wait; destroying the event immediately can race
// (see musa_device.cc: defer destruction until the waiting stream has completed the wait).
struct StreamDepEventCtx {
  musaEvent_t event;
  int device_ordinal;
};

void DestroyStreamDepEventTrampoline(void* p) {
  auto* c = static_cast<StreamDepEventCtx*>(p);
  if (!c) return;
  musaSetDevice(c->device_ordinal);
  (void)musaEventDestroy(c->event);
  delete c;
}

void plugin_se_create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                                      SP_Stream other, TF_Status* status) {
  if (!dependent || !other) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream in create_stream_dependency");
    return;
  }
  musaEvent_t event;
  const int dev_ord = Ordinal(device);
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(dev_ord, status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaEventCreateWithFlags(&event, musaEventDisableTiming);
  if (err != musaSuccess) {
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventCreate");
    return;
  }
  err = musaEventRecord(event, other->stream);
  if (err != musaSuccess) {
    musaEventDestroy(event);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventRecord");
    return;
  }
  err = musaStreamWaitEvent(dependent->stream, event, 0);
  if (err != musaSuccess) {
    musaEventDestroy(event);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamWaitEvent");
    return;
  }
  auto* ctx = new (std::nothrow) StreamDepEventCtx{event, dev_ord};
  if (!ctx) {
    (void)musaStreamSynchronize(dependent->stream);
    musaSetDevice(dev_ord);
    musaEventDestroy(event);
    MallocOrBadAlloc(status);
    return;
  }
  err = musaLaunchHostFunc(dependent->stream, DestroyStreamDepEventTrampoline, ctx);
  if (err != musaSuccess) {
    (void)musaStreamSynchronize(dependent->stream);
    musaSetDevice(dev_ord);
    (void)musaEventDestroy(ctx->event);
    delete ctx;
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaLaunchHostFunc");
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void plugin_se_get_stream_status(const SP_Device* device, SP_Stream stream,
                                 TF_Status* status) {
  if (!stream) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t r = musaStreamQuery(stream->stream);
  if (r == musaSuccess || r == musaErrorNotReady) {
    TF_SetStatus(status, TF_OK, "");
  } else {
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, r, "musaStreamQuery");
  }
}

void plugin_se_create_event(const SP_Device* device, SP_Event* event,
                            TF_Status* status) {
  if (!event) return;
  *event = nullptr;
  auto* e = new (std::nothrow) SP_Event_st();
  if (!e) {
    MallocOrBadAlloc(status);
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) {
    delete e;
    return;
  }
  musaError_t err = musaEventCreateWithFlags(&e->event, musaEventDefault);
  if (err != musaSuccess) {
    delete e;
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventCreate");
    return;
  }
  *event = e;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_se_destroy_event(const SP_Device* device, SP_Event event) {
  if (!event) return;
  TF_Status* st = TF_NewStatus();
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), st);
  if (TF_GetCode(st) == TF_OK) {
    (void)musaEventDestroy(event->event);
  }
  TF_DeleteStatus(st);
  delete event;
}

SE_EventStatus plugin_se_get_event_status(const SP_Device* device, SP_Event event) {
  if (!event) return SE_EVENT_ERROR;
  TF_Status* st = TF_NewStatus();
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), st);
  if (TF_GetCode(st) != TF_OK) {
    TF_DeleteStatus(st);
    return SE_EVENT_ERROR;
  }
  TF_DeleteStatus(st);
  musaError_t r = musaEventQuery(event->event);
  if (r == musaSuccess) return SE_EVENT_COMPLETE;
  if (r == musaErrorNotReady) return SE_EVENT_PENDING;
  return SE_EVENT_ERROR;
}

void plugin_se_record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                            TF_Status* status) {
  if (!stream || !event) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream or event");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaEventRecord(event->event, stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventRecord");
}

void plugin_se_wait_for_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                              TF_Status* status) {
  if (!stream || !event) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream or event");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaStreamWaitEvent(stream->stream, event->event, 0);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamWaitEvent");
}

void plugin_se_create_timer(const SP_Device* device, SP_Timer* timer, TF_Status* status) {
  if (!timer) return;
  *timer = nullptr;
  auto* t = new (std::nothrow) SP_Timer_st();
  if (!t) {
    MallocOrBadAlloc(status);
    return;
  }
  t->has_stop = false;
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) {
    delete t;
    return;
  }
  musaError_t e1 = musaEventCreate(&t->start_event);
  musaError_t e2 = musaEventCreate(&t->end_event);
  if (e1 != musaSuccess || e2 != musaSuccess) {
    if (e1 == musaSuccess) musaEventDestroy(t->start_event);
    if (e2 == musaSuccess) musaEventDestroy(t->end_event);
    delete t;
    TF_SetStatus(status, TF_INTERNAL, "musaEventCreate (timer) failed");
    return;
  }
  *timer = t;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_se_destroy_timer(const SP_Device* device, SP_Timer timer) {
  if (!timer) return;
  TF_Status* st = TF_NewStatus();
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), st);
  if (TF_GetCode(st) == TF_OK) {
    (void)musaEventDestroy(timer->start_event);
    (void)musaEventDestroy(timer->end_event);
  }
  TF_DeleteStatus(st);
  delete timer;
}

void plugin_se_start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                           TF_Status* status) {
  if (!stream || !timer) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream or timer");
    return;
  }
  timer->has_stop = false;
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaEventRecord(timer->start_event, stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventRecord start");
}

void plugin_se_stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                          TF_Status* status) {
  if (!stream || !timer) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream or timer");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaEventRecord(timer->end_event, stream->stream);
  if (err == musaSuccess) {
    timer->has_stop = true;
  }
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventRecord end");
}

void plugin_se_memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status) {
  if (!host_dst || !device_src || !stream) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in memcpy_dtoh");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  const void* src = device_src->opaque;
  musaError_t err = musaMemcpyAsync(host_dst, src, size, musaMemcpyDeviceToHost,
                                    stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpyAsync D2H");
}

void plugin_se_memcpy_htod(const SP_Device* device, SP_Stream stream,
                           SP_DeviceMemoryBase* device_dst, const void* host_src,
                           uint64_t size, TF_Status* status) {
  if (!device_dst || !host_src || !stream) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in memcpy_htod");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  void* dst = device_dst->opaque;
  musaError_t err = musaMemcpyAsync(dst, host_src, size, musaMemcpyHostToDevice,
                                    stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpyAsync H2D");
}

void plugin_se_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                           SP_DeviceMemoryBase* device_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status) {
  if (!device_dst || !device_src || !stream) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in memcpy_dtod");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaMemcpyAsync(device_dst->opaque, device_src->opaque, size,
                                    musaMemcpyDeviceToDevice, stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpyAsync D2D");
}

void plugin_se_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                                const SP_DeviceMemoryBase* device_src, uint64_t size,
                                TF_Status* status) {
  if (!host_dst || !device_src) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in sync_memcpy_dtoh");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err =
      musaMemcpy(host_dst, device_src->opaque, size, musaMemcpyDeviceToHost);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpy D2H");
}

void plugin_se_sync_memcpy_htod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                                const void* host_src, uint64_t size, TF_Status* status) {
  if (!device_dst || !host_src) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in sync_memcpy_htod");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err =
      musaMemcpy(device_dst->opaque, host_src, size, musaMemcpyHostToDevice);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpy H2D");
}

void plugin_se_sync_memcpy_dtod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                                const SP_DeviceMemoryBase* device_src, uint64_t size,
                                TF_Status* status) {
  if (!device_dst || !device_src) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null pointer in sync_memcpy_dtod");
    return;
  }
  if (size == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaMemcpy(device_dst->opaque, device_src->opaque, size,
                               musaMemcpyDeviceToDevice);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpy D2D");
}

void plugin_se_block_host_for_event(const SP_Device* device, SP_Event event,
                                    TF_Status* status) {
  if (!event) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null event");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaEventSynchronize(event->event);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaEventSynchronize");
}

void plugin_se_block_host_until_done(const SP_Device* device, SP_Stream stream,
                                     TF_Status* status) {
  if (!stream) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null stream");
    return;
  }
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaStreamSynchronize(stream->stream);
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamSynchronize");
}

void plugin_se_synchronize_all_activity(const SP_Device* device, TF_Status* status) {
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), status);
  if (TF_GetCode(status) != TF_OK) return;
  musaError_t err = musaDeviceSynchronize();
  ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaDeviceSynchronize");
}

struct HostCbCtx {
  SE_StatusCallbackFn fn;
  void* arg;
};

void HostTrampoline(void* p) {
  auto* c = static_cast<HostCbCtx*>(p);
  TF_Status* s = TF_NewStatus();
  c->fn(c->arg, s);
  TF_DeleteStatus(s);
  delete c;
}

TF_Bool plugin_se_host_callback(const SP_Device* device, SP_Stream stream,
                                      SE_StatusCallbackFn callback_fn, void* callback_arg) {
  if (!stream || !callback_fn) return 0;
  TF_Status* st = TF_NewStatus();
  ::tensorflow::musa::runtime::SetMusaDeviceOrStatus(Ordinal(device), st);
  if (TF_GetCode(st) != TF_OK) {
    TF_DeleteStatus(st);
    return 0;
  }
  TF_DeleteStatus(st);
  auto* ctx = new (std::nothrow) HostCbCtx{callback_fn, callback_arg};
  if (!ctx) return 0;
  musaError_t err = musaLaunchHostFunc(stream->stream, HostTrampoline, ctx);
  if (err != musaSuccess) {
    delete ctx;
    return 0;
  }
  return 1;
}

uint64_t plugin_timer_nanoseconds(SP_Timer timer) {
  if (!timer || !timer->has_stop) return 0;
  float ms = 0;
  musaError_t err = musaEventElapsedTime(&ms, timer->start_event, timer->end_event);
  if (err != musaSuccess) return 0;
  return static_cast<uint64_t>(ms * 1e6f);
}

}  // namespace

// Stable storage if TensorFlow calls SE_InitPlugin with null platform pointers.
static SP_Platform g_musa_se_platform;
static SP_PlatformFns g_musa_se_platform_fns;

extern "C" {

// Shared by plugin_get_device_count and MUSA_TestPluginGetDeviceCount.
static void MUSA_PluginGetDeviceCountBody(int* count, TF_Status* status) {
  int n = 0;
  musaError_t err = musaGetDeviceCount(&n);
  if (err != musaSuccess) {
    if (!::tensorflow::musa::plugin_env::StrictPhysicalDeviceEnum()) {
      VLOG(1) << "musaGetDeviceCount failed; plugin reports 0 devices "
                 "(set MUSA_STRICT_DEVICE_ENUM=1 to treat as error): "
              << musaGetErrorString(err);
      *count = 0;
      TF_SetStatus(status, TF_OK, "");
      return;
    }
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaGetDeviceCount");
    return;
  }
  *count = n;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_get_device_count(const SP_Platform*, int* count, TF_Status* status) {
  if (!count) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null count");
    return;
  }
  MUSA_PluginGetDeviceCountBody(count, status);
}

// Test/CI: same behavior as `plugin_get_device_count` (no `SP_Platform*`
// indirection) so Python can assert strict vs non-strict `musaGetDeviceCount`
// without loading TensorFlow Pluggable full stack.
void MUSA_TestPluginGetDeviceCount(int* count, TF_Status* status) {
  if (!count) {
    if (status) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "null count");
    }
    return;
  }
  if (!status) {
    return;
  }
  MUSA_PluginGetDeviceCountBody(count, status);
}

// CI/driver smoke: set device, stream, tiny H2D/D2H copy. Does not require TF.
// Relaxed enumeration (`relaxed_enum=true`): when `musaGetDeviceCount` fails,
// returns TF_OK with a skip message so no-hardware CI stays green.
// Strict (`relaxed_enum=false`): `musaGetDeviceCount` failure propagates as an error;
// zero GPUs returns FAILED_PRECONDITION so `TF_OK` reserved for successful memcpy
// verification (avoid false-positive `TF_OK` on CI without hardware).

namespace {
void MUSA_SeRuntimeMemcpySmokeImpl(TF_Status* status, bool relaxed_enum) {
  if (!status) {
    return;
  }
  int n = 0;
  musaError_t err = musaGetDeviceCount(&n);
  if (err != musaSuccess) {
    if (relaxed_enum) {
      TF_SetStatus(status, TF_OK,
                   "MUSA_TestSeRuntimeSmoke: musaGetDeviceCount failed (smoke skipped)");
      return;
    }
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaGetDeviceCount");
    return;
  }
  if (n <= 0) {
    if (relaxed_enum) {
      TF_SetStatus(status, TF_OK, "MUSA_TestSeRuntimeSmoke: zero devices (smoke skipped)");
      return;
    }
    TF_SetStatus(
        status, TF_FAILED_PRECONDITION,
        "MUSA_TestSeRuntimeSmokeStrict: zero MUSA devices (strict smoke skipped)");
    return;
  }
  err = musaSetDevice(0);
  if (err != musaSuccess) {
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaSetDevice");
    return;
  }
  musaStream_t stream = nullptr;
  err = musaStreamCreate(&stream);
  if (err != musaSuccess) {
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamCreate");
    return;
  }
  uint32_t host_in = 0xAABBCCDD;
  void* d = nullptr;
  err = musaMalloc(&d, sizeof(uint32_t));
  if (err != musaSuccess) {
    (void)musaStreamDestroy(stream);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMalloc");
    return;
  }
  err = musaMemcpyAsync(d, &host_in, sizeof(uint32_t), musaMemcpyHostToDevice, stream);
  if (err != musaSuccess) {
    (void)musaFree(d);
    (void)musaStreamDestroy(stream);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpyAsync H2D");
    return;
  }
  uint32_t host_out = 0;
  err = musaMemcpyAsync(&host_out, d, sizeof(uint32_t), musaMemcpyDeviceToHost, stream);
  if (err != musaSuccess) {
    (void)musaFree(d);
    (void)musaStreamDestroy(stream);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaMemcpyAsync D2H");
    return;
  }
  err = musaStreamSynchronize(stream);
  if (err != musaSuccess) {
    (void)musaFree(d);
    (void)musaStreamDestroy(stream);
    ::tensorflow::musa::runtime::SetStatusFromMusa(status, err, "musaStreamSynchronize");
    return;
  }
  (void)musaFree(d);
  (void)musaStreamDestroy(stream);
  if (host_out != host_in) {
    TF_SetStatus(status, TF_INTERNAL, "MUSA_TestSeRuntimeSmoke: memcpy check failed");
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}
}  // namespace

void MUSA_TestSeRuntimeSmoke(TF_Status* status) {
  MUSA_SeRuntimeMemcpySmokeImpl(status, /*relaxed_enum=*/true);
}

void MUSA_TestSeRuntimeSmokeStrict(TF_Status* status) {
  MUSA_SeRuntimeMemcpySmokeImpl(status, /*relaxed_enum=*/false);
}

size_t MUSA_TestSeRegistryLiveOrdinalsForTest() {
  return ::tensorflow::musa::MusaSeRegistrySizeForTest();
}

void MUSA_TestRegistryDeviceLifecycle(TF_Status* status) {
  if (!status) {
    return;
  }
  ::tensorflow::musa::MusaSeRegistryResetForTest();
  if (::tensorflow::musa::MusaSeRegistrySizeForTest() != 0) {
    TF_SetStatus(status, TF_INTERNAL,
                 "MUSA_TestRegistryDeviceLifecycle: registry not empty after reset");
    return;
  }
  constexpr int32_t kTestOrdinal = 42;
  ::tensorflow::musa::MusaSeRegistryOnDeviceCreated(kTestOrdinal);
  if (::tensorflow::musa::MusaSeRegistrySizeForTest() != 1) {
    ::tensorflow::musa::MusaSeRegistryResetForTest();
    TF_SetStatus(status, TF_INTERNAL,
                 "MUSA_TestRegistryDeviceLifecycle: expected one live ordinal");
    return;
  }
  ::tensorflow::musa::MusaSeRegistryOnDeviceDestroyed(kTestOrdinal);
  if (::tensorflow::musa::MusaSeRegistrySizeForTest() != 0) {
    ::tensorflow::musa::MusaSeRegistryResetForTest();
    TF_SetStatus(status, TF_INTERNAL,
                 "MUSA_TestRegistryDeviceLifecycle: expected empty registry");
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void plugin_create_device(const SP_Platform*, SE_CreateDeviceParams* params,
                          TF_Status* status) {
  if (!params || !params->device) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null create_device params");
    return;
  }
  auto* dev = params->device;
  dev->struct_size = SP_DEVICE_STRUCT_SIZE;
  dev->ext = nullptr;
  dev->ordinal = params->ordinal;
  // Own a copy of ordinal for get_device
  int* h = new (std::nothrow) int(params->ordinal);
  if (!h) {
    MallocOrBadAlloc(status);
    return;
  }
  dev->device_handle = h;
  dev->hardware_name = kPlatformName;
  dev->device_vendor = kVendorMthreads;
  dev->pci_bus_id = nullptr;
  ::tensorflow::musa::MusaSeRegistryOnDeviceCreated(
      static_cast<int32_t>(params->ordinal));
  TF_SetStatus(status, TF_OK, "");
}

void plugin_destroy_device(const SP_Platform*, SP_Device* device) {
  if (!device) return;
  const int32_t ord = static_cast<int32_t>(device->ordinal);
  ::tensorflow::musa::MusaSeRegistryOnDeviceDestroyed(ord);
  if (device->device_handle) {
    delete static_cast<int*>(device->device_handle);
    device->device_handle = nullptr;
  }
}

int32_t plugin_get_numa_node(const SP_Device*) { return -1; }

int64_t plugin_get_memory_bandwidth(const SP_Device*) { return -1; }

double plugin_get_gflops(const SP_Device*) { return -1.0; }

void plugin_create_device_fns(const SP_Platform*, SE_CreateDeviceFnsParams* params,
                                TF_Status* status) {
  if (!params || !params->device_fns) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null create_device_fns");
    return;
  }
  SP_DeviceFns* f = params->device_fns;
  f->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  f->ext = nullptr;
  f->get_numa_node = plugin_get_numa_node;
  f->get_memory_bandwidth = plugin_get_memory_bandwidth;
  f->get_gflops = plugin_get_gflops;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_destroy_device_fns(const SP_Platform*, SP_DeviceFns*) {}

void plugin_create_stream_executor(const SP_Platform*, SE_CreateStreamExecutorParams* params,
                                   TF_Status* status) {
  if (!params || !params->stream_executor) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null create_stream_executor");
    return;
  }
  SP_StreamExecutor* se = params->stream_executor;
  se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  se->ext = nullptr;
  se->allocate = plugin_se_allocate;
  se->deallocate = plugin_se_deallocate;
  se->host_memory_allocate = plugin_se_host_memory_allocate;
  se->host_memory_deallocate = plugin_se_host_memory_deallocate;
  se->unified_memory_allocate = plugin_se_unified_memory_allocate;
  se->unified_memory_deallocate = plugin_se_unified_memory_deallocate;
  se->get_allocator_stats = plugin_se_get_allocator_stats;
  se->device_memory_usage = plugin_se_device_memory_usage;
  se->create_stream = plugin_se_create_stream;
  se->destroy_stream = plugin_se_destroy_stream;
  se->create_stream_dependency = plugin_se_create_stream_dependency;
  se->get_stream_status = plugin_se_get_stream_status;
  se->create_event = plugin_se_create_event;
  se->destroy_event = plugin_se_destroy_event;
  se->get_event_status = plugin_se_get_event_status;
  se->record_event = plugin_se_record_event;
  se->wait_for_event = plugin_se_wait_for_event;
  se->create_timer = plugin_se_create_timer;
  se->destroy_timer = plugin_se_destroy_timer;
  se->start_timer = plugin_se_start_timer;
  se->stop_timer = plugin_se_stop_timer;
  se->memcpy_dtoh = plugin_se_memcpy_dtoh;
  se->memcpy_htod = plugin_se_memcpy_htod;
  se->memcpy_dtod = plugin_se_memcpy_dtod;
  se->sync_memcpy_dtoh = plugin_se_sync_memcpy_dtoh;
  se->sync_memcpy_htod = plugin_se_sync_memcpy_htod;
  se->sync_memcpy_dtod = plugin_se_sync_memcpy_dtod;
  se->block_host_for_event = plugin_se_block_host_for_event;
  se->block_host_until_done = plugin_se_block_host_until_done;
  se->synchronize_all_activity = plugin_se_synchronize_all_activity;
  se->host_callback = plugin_se_host_callback;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_destroy_stream_executor(const SP_Platform*, SP_StreamExecutor*) {}

void plugin_create_timer_fns(const SP_Platform*, SP_TimerFns* timer, TF_Status* status) {
  if (!timer) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null timer fns");
    return;
  }
  timer->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
  timer->ext = nullptr;
  timer->nanoseconds = plugin_timer_nanoseconds;
  TF_SetStatus(status, TF_OK, "");
}

void plugin_destroy_timer_fns(const SP_Platform*, SP_TimerFns*) {}

void plugin_destroy_platform(SP_Platform*) {}

void plugin_destroy_platform_fns(SP_PlatformFns*) {}

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  if (!params) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "null SE_PlatformRegistrationParams");
    return;
  }
  if (params->struct_size == 0) {
    params->struct_size = SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE;
  } else if (params->struct_size < SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "SE_PlatformRegistrationParams::struct_size too small for this "
                 "TensorFlow build");
    return;
  }
  if (!::tensorflow::musa::plugin_env::PluggableSePathEnabled()) {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        "MUSA: StreamExecutor Pluggable path is not enabled. Use "
        "tf.load_op_library / import tensorflow_musa (default), or set "
        "MUSA_ENABLE_SE_PLUGIN=1 for tensorflow-plugins/SE_InitPlugin-only "
        "mode (disables the legacy C++ MUSA device registration in this .so).");
    return;
  }
  if (!params->platform) {
    std::memset(&g_musa_se_platform, 0, sizeof(g_musa_se_platform));
    params->platform = &g_musa_se_platform;
  }
  if (!params->platform_fns) {
    std::memset(&g_musa_se_platform_fns, 0, sizeof(g_musa_se_platform_fns));
    params->platform_fns = &g_musa_se_platform_fns;
  }
  params->major_version = SE_MAJOR;
  params->minor_version = SE_MINOR;
  params->patch_version = SE_PATCH;
  params->destroy_platform = plugin_destroy_platform;
  params->destroy_platform_fns = plugin_destroy_platform_fns;

  SP_Platform* p = params->platform;
  p->struct_size = SP_PLATFORM_STRUCT_SIZE;
  p->ext = nullptr;
  p->name = kPlatformName;
  p->type = kPlatformType;
  p->supports_unified_memory = 0;
  // Align with PluggableDevice stack: let TensorFlow wrap the plugin allocator
  // with a BFC when the plugin chooses raw musaMalloc in allocate().
  p->use_bfc_allocator = 1;

  SP_PlatformFns* pf = params->platform_fns;
  pf->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
  pf->ext = nullptr;
  pf->get_device_count = plugin_get_device_count;
  pf->create_device = plugin_create_device;
  pf->destroy_device = plugin_destroy_device;
  pf->create_device_fns = plugin_create_device_fns;
  pf->destroy_device_fns = plugin_destroy_device_fns;
  pf->create_stream_executor = plugin_create_stream_executor;
  pf->destroy_stream_executor = plugin_destroy_stream_executor;
  pf->create_timer_fns = plugin_create_timer_fns;
  pf->destroy_timer_fns = plugin_destroy_timer_fns;

  TF_SetStatus(status, TF_OK, "");
}

}  // extern "C"
