# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Memory snapshot & history toolchain (plan §5.6 / §6.3 optional tier).

Three public entry points:

* :func:`memory_snapshot` — point-in-time structured dict describing
  everything observable about the caching allocator (stats, segments,
  driver info, allocator config, backend, and — when recording is on —
  a sampled time-series of ``memory_stats``). Pure-Python composition of
  existing ``_C`` probes; no new native hooks needed.
* :func:`_dump_snapshot` — JSON serialization of the above to disk;
  handy as the deliverable for bug reports / post-mortem OOM triage.
* :func:`_record_memory_history` — flip a lightweight background
  sampler on/off. It polls :func:`memory_stats` at a user-configurable
  cadence and keeps the last N samples in a ring buffer. We deliberately
  sample rather than hook every alloc/free so the feature costs nothing
  in the hot path and stays on the native side of TF's stability
  contract (no allocator-callback ABI to version).

The plan §4.2 S5 note calls out that symbolicated stack traces would
require libbacktrace and are out of scope for the first drop; this
module keeps that promise — the snapshot carries sizes / counts /
timestamps only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from ._ext import _C

__all__ = [
    "memory_snapshot",
    "_dump_snapshot",
    "_record_memory_history",
]


def _resolve_ordinal(device: Optional[int]) -> int:
    """Minimal mirror of memory._resolve_ordinal.

    Duplicated rather than imported so this module stays usable on its
    own (some tests exercise snapshot.py without loading memory.py).
    """
    if device is None:
        return 0
    if isinstance(device, bool) or not isinstance(device, int):
        raise TypeError(
            f"device must be an int or None, got {type(device).__name__}"
        )
    if device < 0:
        raise ValueError(f"device ordinal must be >= 0, got {device}")
    return device


# ---------------------------------------------------------------------------
# History sampler — one background thread per process.
# ---------------------------------------------------------------------------


class _HistoryRecorder:
    """Polling-based recorder for ``memory_stats`` time-series.

    Design notes:
    * Sampling (instead of alloc/free hooks) keeps the native ABI flat
      and costs O(1) per tick; over-polling is harmless because we only
      snapshot counters, not the allocator's internal state.
    * The thread is a daemon so it never blocks interpreter shutdown —
      users who forget to turn recording off won't see hangs.
    * ``_lock`` only guards config + buffer swaps; readers that want a
      point-in-time copy take the lock briefly and walk the list by
      slice, which is O(N) in the buffer size but fine for N <= 10_000.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._buffer: List[Dict[str, Any]] = []
        self._max_entries: int = 0
        self._interval_s: float = 0.05
        self._device: int = 0
        self._started_at_ns: int = 0

    def start(
        self,
        *,
        device: int,
        max_entries: int,
        interval_ms: int,
    ) -> None:
        # Restart cleanly if the caller flipped the knobs.
        self.stop()
        with self._lock:
            self._buffer = []
            self._max_entries = max(1, int(max_entries))
            self._interval_s = max(0.001, float(interval_ms) / 1000.0)
            self._device = int(device)
            self._stop = threading.Event()
            self._started_at_ns = time.monotonic_ns()
            t = threading.Thread(
                target=self._run,
                name="tensorflow_musa-history-sampler",
                daemon=True,
            )
            self._thread = t
            t.start()

    def stop(self) -> None:
        with self._lock:
            t = self._thread
            self._thread = None
            if t is not None:
                self._stop.set()
        if t is not None:
            # Join outside the lock so the worker can acquire it if
            # needed while draining.
            t.join(timeout=1.0)

    def is_active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def entries(self) -> List[Dict[str, Any]]:
        # Return a shallow copy; individual entries are plain dicts and
        # never mutated after being appended, so aliasing them is safe.
        with self._lock:
            return list(self._buffer)

    def _run(self) -> None:
        interval = self._interval_s
        max_n = self._max_entries
        device = self._device
        while not self._stop.is_set():
            try:
                sample = _C._device_allocator_stats(device)
                sample["t_ns"] = time.monotonic_ns() - self._started_at_ns
            except Exception:
                # The sampler must never take down the host process;
                # if the allocator isn't ready yet we just skip this
                # tick and try again next interval.
                self._stop.wait(interval)
                continue
            with self._lock:
                self._buffer.append(sample)
                if len(self._buffer) > max_n:
                    # Ring-buffer trim: keep the newest `max_n` entries.
                    # Slicing rather than popleft because we expose
                    # `list` semantics and typical N is small.
                    del self._buffer[: len(self._buffer) - max_n]
            self._stop.wait(interval)


_recorder = _HistoryRecorder()


def _record_memory_history(
    enabled: bool,
    *,
    max_entries: int = 1024,
    interval_ms: int = 50,
    device: Optional[int] = None,
) -> None:
    """Toggle the memory-history background sampler.

    Args:
        enabled: ``True`` to start recording (replacing any prior
            session). ``False`` to stop and discard the thread; the
            buffer is kept so a subsequent :func:`memory_snapshot`
            call can still drain the last N samples.
        max_entries: Ring-buffer capacity; oldest entries are evicted
            first. Must be a positive integer.
        interval_ms: Polling cadence in milliseconds. Values below 1
            are clamped to 1ms to protect the sampler from busy-looping.
        device: Device ordinal to monitor (defaults to 0).

    The leading underscore mirrors torch.musa's naming: the API is
    stable enough to document but we reserve the right to add columns
    to each sample as the allocator grows more telemetry.
    """
    if enabled:
        _recorder.start(
            device=_resolve_ordinal(device),
            max_entries=max_entries,
            interval_ms=interval_ms,
        )
    else:
        _recorder.stop()


# ---------------------------------------------------------------------------
# Snapshot composition — reads every available _C probe.
# ---------------------------------------------------------------------------


def memory_snapshot(device: Optional[int] = None) -> Dict[str, Any]:
    """Return a structured snapshot of the caching allocator's state.

    The return shape is:

    .. code-block:: python

        {
            "device": 0,
            "backend": "caching",
            "timestamp_ns": <int, time.time_ns at capture>,
            "stats": {...},            # _device_allocator_stats
            "segments": [{...}, ...],  # _device_segment_snapshot
            "driver": {"free_bytes": int, "total_bytes": int},
            "allocator_config": {...}, # TF_MUSA_ALLOC_CONF parsed view
            "vmm": {"available": bool, "supported": bool,
                    "granularity_bytes": int},
            "last_oom_message": "",    # _device_last_oom_message
            "history": {               # present only while recording
                "active": bool,
                "entries": [{...}, ...],
            },
        }

    Callers treat missing keys as "feature not available in this build"
    rather than "value is zero"; this lets us add fields in a backward-
    compatible way.
    """
    ordinal = _resolve_ordinal(device)
    snap: Dict[str, Any] = {
        "device": ordinal,
        "backend": _C._device_allocator_backend(),
        "timestamp_ns": time.time_ns(),
        "stats": dict(_C._device_allocator_stats(ordinal)),
        "segments": [dict(s) for s in _C._device_segment_snapshot(ordinal)],
        "allocator_config": dict(_C._allocator_config()),
        "last_oom_message": _C._device_last_oom_message(ordinal),
    }

    # Driver info is best-effort: it requires an initialized MUSA
    # device and will raise RuntimeError on CPU-only hosts. Snapshots
    # should still be useful in that case (stats stay zeroed-but-shaped),
    # so we swallow the error and mark the field unavailable.
    try:
        free_b, total_b = _C._device_memory_usage(ordinal)
        snap["driver"] = {
            "free_bytes": int(free_b),
            "total_bytes": int(total_b),
        }
    except Exception as exc:
        snap["driver"] = {"error": str(exc)}

    try:
        snap["vmm"] = {
            "available": bool(_C._vmm_available()),
            "supported": bool(_C._vmm_supported(ordinal)),
            "granularity_bytes": int(_C._vmm_granularity(ordinal)),
        }
    except Exception as exc:
        snap["vmm"] = {"error": str(exc)}

    entries = _recorder.entries()
    if entries or _recorder.is_active():
        snap["history"] = {
            "active": _recorder.is_active(),
            "entries": entries,
        }

    return snap


def _dump_snapshot(
    path: str,
    device: Optional[int] = None,
    *,
    indent: Optional[int] = 2,
) -> str:
    """Write :func:`memory_snapshot` to ``path`` as JSON.

    Writes via a ``<path>.tmp`` + ``os.replace`` pair so a crash mid-
    write leaves the original file (if any) untouched — useful when
    dashboards point a file watcher at the output location.

    Args:
        path: Destination filename. Parent directory must exist.
        device: Device ordinal (defaults to 0).
        indent: JSON pretty-print indent; ``None`` for compact output.

    Returns:
        The absolute path that was written.
    """
    snap = memory_snapshot(device)
    abs_path = os.path.abspath(path)
    tmp = abs_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=indent, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, abs_path)
    return abs_path
