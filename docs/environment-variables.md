# Environment Variables

This document lists every `TF_MUSA_*` environment variable recognized by `tensorflow_musa_extension`, organized by subsystem.

## Conventions

- Boolean variables accept `1` / `true` / `on` / `yes` (case-insensitive) as "enable"; everything else (including unset or empty) is "disable".
- Byte-size values accept raw bytes or a `kb` / `mb` / `gb` suffix (e.g. `256mb`). Parsing is case-insensitive.
- "Default" means the value used when the variable is unset or empty.

---

## Device memory allocator

### `TF_MUSA_DEVICE_ALLOCATOR`

- **Default**: `caching`
- **Values**: `caching` | `passthrough`

Selects the device-side allocator backend.

- `caching` — the extension's own best-fit caching allocator on top of `musaMalloc` / `musaFree`, with split / merge and optional VMM expandable segments.
- `passthrough` — every allocation is a direct `musaMalloc` call. Primarily for A/B bisection when investigating allocator-related issues.

### `TF_MUSA_ALLOC_CONF`

- **Default**: empty
- **Format**: comma-separated `key:value` (or `key=value`) pairs. Unknown keys emit a warning and are ignored.

| key | type | description |
|---|---|---|
| `expandable_segments` | bool | Enable VMM-backed expandable segments to reduce fragmentation on long-running workloads. |
| `max_split_size_mb` | int (MiB) | Blocks larger than this are not split. |
| `max_split_size_bytes` | bytes | Same semantic as `max_split_size_mb` but with explicit unit (`1024`, `512kb`, `256mb`, `1gb`). |
| `roundup_power2_divisions` | int | Power-of-two rounding granularity for allocation requests. |
| `garbage_collection_threshold` | float in `(0, 1]` | Reserved for future use. |

Example:

```bash
TF_MUSA_ALLOC_CONF="expandable_segments:true,max_split_size_mb:512"
```

### `TF_MUSA_DEVICE_ALLOC_MAX_POOL_MB`

- **Default**: `32768` (32 GiB)
- **Values**: integer MiB; `0` or any invalid input is treated as unlimited.

Soft cap on the allocator's total reserved bytes. Live allocations above the cap are still honored.

### `TF_MUSA_ALLOC_VERBOSE_OOM`

- **Default**: `0`

When set, OOM paths mirror a structured diagnostic (per-size-bucket counts, largest free block, segment list) to stderr. The diagnostic is always available via the Python API regardless of this flag.

### `TF_MUSA_ENABLE_ASYNC_ALLOC`

- **Default**: `0`

When set, enables the runtime's mempool / `musaMallocAsync` path. Mutually exclusive with the caching allocator's own pool management — some muDNN kernels are known to crash when mixed with the async pool. Keep off unless your workload is known to be safe.

---

## Host pinned-memory allocator

### `TF_MUSA_HOST_ALLOC_MAX_POOL_MB`

- **Default**: `2048`

Upper bound on the host caching allocator's pinned pool (MiB). When exceeded, new allocations still succeed but are not cached; hit-rate drops accordingly.

### `TF_MUSA_DISABLE_HOST_CACHING`

- **Default**: `0`

When set, bypasses the host caching allocator and calls `musaHostAlloc` / `musaFreeHost` directly. A/B bisection only.

---

## H2D staging pool

### `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES`

- **Default**: `0` (disabled)

When `> 0`, H2D copies from a pageable source ≥ this many bytes are staged through a pinned buffer to unlock async overlap. The variable also registers an `atexit` handler that prints pool statistics to stderr:

```
[MUSA] H2D staging stats: staged=N (X MiB) already_pinned=P pool_allocs=A
```

- `staged` — number of copies routed through the staging path.
- `already_pinned` — copies whose source was already pinned; staging is skipped to avoid a redundant host memcpy.
- `pool_allocs` — fresh `musaHostAlloc` calls observed by the pool (cache misses).

Recommended values: `524288` (512 KiB) or `1048576` (1 MiB). Lower thresholds add CPU-memcpy overhead on small payloads.

The staging pool shares its pinned buffers with the host caching allocator, so `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` governs both.

### `TF_MUSA_H2D_STAGING_MEMCPY_THREADS`

- **Default**: `4`
- **Range**: `[1, 16]`

Worker threads used for the pageable → pinned memcpy inside the staging path. Helpful for large payloads (≥ 1 MiB).

### `TF_MUSA_H2D_STAGING_SKIP_MEMCPY`

- **Default**: `0`

Debug only. When set, skips the pageable → pinned memcpy and issues the async H2D directly from the (uninitialized) pinned staging buffer. Use it to measure pool/event overhead in isolation — contents arriving on device are undefined.

### `TF_MUSA_H2D_STAGING_MAX_POOL_MB`

- **Deprecated.** The staging pool is served by the host caching allocator; use `TF_MUSA_HOST_ALLOC_MAX_POOL_MB` instead. Setting this variable emits a one-time warning.

---

## H2D automatic pinning (optional)

### `TF_MUSA_AUTO_PIN_H2D_THRESHOLD_BYTES`

- **Default**: `0` (disabled)

When `> 0`, H2D source buffers ≥ this many bytes are auto-registered with `musaHostRegister` so subsequent copies go through the native async-H2D path without an intermediate staging memcpy.

**Only useful when the same host address is reused repeatedly** (e.g. long-lived `tf.data` buffers). Per-step throw-away NumPy arrays will pay registration cost for no gain.

### `TF_MUSA_DIAG_H2D_PINNED`

- **Default**: `0`

Debug only. Prints the pinned status (`pointerType`, `isPinned`) of the first 50 H2D source buffers to stderr. Useful for confirming whether `feed_dict` inputs arrive as pageable memory.

---

## Event pool

### `TF_MUSA_DISABLE_EVENT_POOL`

- **Default**: `0`

When set, disables the per-device `musaEvent_t` reuse pool and falls back to `musaEventCreateWithFlags` / `musaEventDestroy` for each event. A/B bisection only.

---

## Graph optimizer

### `TF_MUSA_DISABLE_HOST_COMPUTE_PIN`

- **Default**: `0`

When set, disables the `PinHostComputeToCpu` optimizer pass that keeps shape/index computations on CPU. Disabling typically slows things down; use only when investigating placement-related issues.

---

## Quick reference

| Goal | Variable(s) |
|---|---|
| Reduce fragmentation on long-running services | `TF_MUSA_ALLOC_CONF=expandable_segments:true` |
| Async overlap for large pageable H2D feeds | `TF_MUSA_H2D_STAGING_THRESHOLD_BYTES=1048576` |
| Cap per-process device memory | `tensorflow_musa.memory.set_per_process_memory_fraction(f)` (Python API) |
| Diagnose OOM | `TF_MUSA_ALLOC_VERBOSE_OOM=1` |
| Bisect allocator-related issues | `TF_MUSA_DEVICE_ALLOCATOR=passthrough` + `TF_MUSA_DISABLE_HOST_CACHING=1` |
| Confirm H2D source is pageable vs. pinned | `TF_MUSA_DIAG_H2D_PINNED=1` |

Complementary runtime controls are available via the `tensorflow_musa.memory` and `tensorflow_musa.device` Python APIs — `memory_stats`, `memory_snapshot`, `empty_cache`, `set_per_process_memory_fraction`, `can_access_peer`, `enable_peer_access`, and so on.
