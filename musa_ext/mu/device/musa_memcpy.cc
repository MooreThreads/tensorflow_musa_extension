#include "mu/device/musa_memcpy.h"

#include <musa_runtime.h>
#include <stdio.h>

#include "mu/device/musa_telemetry.h"

// The telemetry prologue (`musaGetDevice` + `MUSA_TELEMETRY_ON_MEMCPY`) is
// only meaningful when the tracing subsystem is compiled in. In release
// builds (MUSA_DISABLE_TRACE_LOGGING), the macro is a no-op, so the
// surrounding `musaGetDevice` call is pure overhead on a hot path that
// kernels (AddN, Concat, ResourceVariable assign, etc.) walk many times
// per step. Guard the whole block to drop the driver round-trip entirely.
#ifdef MUSA_DISABLE_TRACE_LOGGING
#define MUSA_MEMCPY_TELEMETRY_PROLOGUE(dst, src, size, stream_id, type) \
  do {                                                                  \
  } while (0)
#else
#define MUSA_MEMCPY_TELEMETRY_PROLOGUE(dst, src, size, stream_id, type)        \
  do {                                                                         \
    int _dev = -1;                                                             \
    musaGetDevice(&_dev);                                                      \
    MUSA_TELEMETRY_ON_MEMCPY((dst),                                            \
                             const_cast<void*>(static_cast<const void*>(src)), \
                             (size), _dev, (stream_id), (type));               \
  } while (0)
#endif

namespace tensorflow {
namespace musa {

// Helper function to get a synchronous stream for blocking operations
static musaStream_t GetSynchronousStream() {
  // Use the default stream (0) for synchronous operations
  // This ensures proper synchronization without creating extra streams
  return 0;
}

mStatus MusaMemcpyD2H(void* h, const void* d, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (h == nullptr || d == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyD2H failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            h, d, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(h, d, size, 0, TelemetryEventType::kMemcpyD2H);

  // Use async copy with immediate synchronization for better performance
  // and to allow potential optimizations in the driver
  musaStream_t sync_stream = GetSynchronousStream();
  musaError_t err =
      musaMemcpyAsync(h, d, size, musaMemcpyDeviceToHost, sync_stream);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsync D2H failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), h, d, size);
    return static_cast<mStatus>(1);
  }

  // Synchronize to ensure completion
  err = musaStreamSynchronize(sync_stream);
  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyD2H stream sync failed: %s\n",
            musaGetErrorString(err));
    return static_cast<mStatus>(1);
  }

  return mStatus::SUCCESS;
}

mStatus MusaMemcpyH2D(void* d, const void* h, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyH2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            d, h, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(d, h, size, 0, TelemetryEventType::kMemcpyH2D);

  // Use async copy with immediate synchronization
  musaStream_t sync_stream = GetSynchronousStream();
  musaError_t err =
      musaMemcpyAsync(d, h, size, musaMemcpyHostToDevice, sync_stream);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsync H2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d, h, size);
    return static_cast<mStatus>(1);
  }

  // Synchronize to ensure completion
  err = musaStreamSynchronize(sync_stream);
  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyH2D stream sync failed: %s\n",
            musaGetErrorString(err));
    return static_cast<mStatus>(1);
  }

  return mStatus::SUCCESS;
}

mStatus MusaMemcpyD2D(void* d1, const void* d2, size_t size) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyD2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            d1, d2, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(d1, d2, size, 0,
                                 TelemetryEventType::kMemcpyD2D);

  // For D2D, we can use the default stream since it's device-local
  musaStream_t sync_stream = GetSynchronousStream();
  musaError_t err =
      musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, sync_stream);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsync D2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d1, d2, size);
    return static_cast<mStatus>(1);
  }

  // Synchronize to ensure completion
  err = musaStreamSynchronize(sync_stream);
  if (err != musaSuccess) {
    fprintf(stderr, "[MUSA] ERROR: MusaMemcpyD2D stream sync failed: %s\n",
            musaGetErrorString(err));
    return static_cast<mStatus>(1);
  }

  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2H(void* h, const void* d, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (h == nullptr || d == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncD2H failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            h, d, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(h, d, size, MUSA_TELEMETRY_STREAM_ID(s),
                                 TelemetryEventType::kMemcpyD2H);

  musaError_t err = musaMemcpyAsync(h, d, size, musaMemcpyDeviceToHost, s);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncD2H failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), h, d, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncH2D(void* d, const void* h, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d == nullptr || h == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncH2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            d, h, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(d, h, size, MUSA_TELEMETRY_STREAM_ID(s),
                                 TelemetryEventType::kMemcpyH2D);

  musaError_t err = musaMemcpyAsync(d, h, size, musaMemcpyHostToDevice, s);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncH2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d, h, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

mStatus MusaMemcpyAsyncD2D(void* d1, const void* d2, size_t size,
                           musaStream_t s) {
  if (size == 0) {
    return mStatus::SUCCESS;
  }
  if (d1 == nullptr || d2 == nullptr) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncD2D failed: null pointer "
            "(dst=%p, src=%p, size=%zu)\n",
            d1, d2, size);
    return static_cast<mStatus>(1);
  }

  MUSA_MEMCPY_TELEMETRY_PROLOGUE(d1, d2, size, MUSA_TELEMETRY_STREAM_ID(s),
                                 TelemetryEventType::kMemcpyD2D);

  musaError_t err = musaMemcpyAsync(d1, d2, size, musaMemcpyDeviceToDevice, s);

  if (err != musaSuccess) {
    fprintf(stderr,
            "[MUSA] ERROR: MusaMemcpyAsyncD2D failed: %s "
            "(dst=%p, src=%p, size=%zu)\n",
            musaGetErrorString(err), d1, d2, size);
    return static_cast<mStatus>(1);
  }
  return mStatus::SUCCESS;
}

}  // namespace musa
}  // namespace tensorflow
