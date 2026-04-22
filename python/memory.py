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

"""MUSA memory introspection and control APIs (plan §6.3).

This module intentionally exposes only the subset of torch.musa.memory
that covers diagnosis and coarse-grained tuning: allocator stats,
manual cache drain, per-process memory limit, and raw driver info.
Higher-level constructs (Stream, Event, MemPool, record_memory_history)
are out of scope for the initial drop — see the plan's "明确不做"
section for the rationale.

All functions accept an optional ``device`` ordinal (defaulting to 0,
which matches the behavior documented by ``torch.musa.memory``); the
``None`` sentinel is accepted as "use the default device" for parity
with PyTorch's shorter call sites. Counts and byte totals are always
returned as Python ``int`` so users can compare them directly to the
values printed by external tooling like ``nvidia-smi``.
"""

from typing import Dict, Optional, Tuple

from ._ext import _C
from .snapshot import _dump_snapshot, _record_memory_history, memory_snapshot

__all__ = [
    "empty_cache",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "reset_peak_memory_stats",
    "memory_stats",
    "set_per_process_memory_fraction",
    "mem_get_info",
    "get_allocator_backend",
    # Snapshot / history — optional diagnostic tier (plan §6.3).
    "memory_snapshot",
    "_dump_snapshot",
    "_record_memory_history",
]


def _resolve_ordinal(device: Optional[int]) -> int:
    """Normalize user-supplied device args.

    ``None`` → 0 for parity with torch.musa's convenience overloads.
    Non-integer values raise early so the underlying C extension
    isn't asked to cope with surprises.
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


def empty_cache(device: Optional[int] = None) -> int:
    """Release cached-but-idle device memory back to the driver.

    Scans every fully-free segment held by the caching allocator on
    ``device`` and returns them via ``musaFree`` (or the VMM
    unmap/release path when ``TF_MUSA_ALLOC_CONF=expandable_segments:true``
    is active). Live allocations are never touched.

    Useful before a large one-off allocation (e.g. loading a
    checkpoint) or as the response to an OOM to let the next request
    try the driver with a clean slate.

    Args:
        device: Device ordinal (defaults to 0).

    Returns:
        The number of bytes returned to the driver.
    """
    return int(_C._device_empty_cache(_resolve_ordinal(device)))


def memory_allocated(device: Optional[int] = None) -> int:
    """Return bytes currently handed out to callers by the allocator.

    Reflects the sum of every live allocation; does not include the
    allocator's free cache. Matches ``torch.musa.memory_allocated``.
    """
    return int(_C._device_allocator_stats(_resolve_ordinal(device))["in_use_bytes"])


def memory_reserved(device: Optional[int] = None) -> int:
    """Return bytes currently held by the allocator (live + cached).

    This is the driver-visible footprint of the caching allocator.
    When ``expandable_segments`` is active it counts VMM-mapped
    bytes too. Matches ``torch.musa.memory_reserved``.
    """
    return int(_C._device_allocator_stats(_resolve_ordinal(device))["reserved_bytes"])


def max_memory_allocated(device: Optional[int] = None) -> int:
    """Return the peak ``memory_allocated`` value since last reset.

    Peak tracking spans the process lifetime; call
    :func:`reset_peak_memory_stats` to rebase after a warm-up phase.
    """
    return int(
        _C._device_allocator_stats(_resolve_ordinal(device))["peak_in_use_bytes"]
    )


def reset_peak_memory_stats(device: Optional[int] = None) -> None:
    """Rebase ``max_memory_allocated`` to the current ``memory_allocated``.

    Typical use: call after warmup steps so the peak measurement
    reflects steady-state-training behavior rather than one-off
    initialization spikes.
    """
    _C._device_reset_peak_stats(_resolve_ordinal(device))


def memory_stats(device: Optional[int] = None) -> Dict[str, int]:
    """Return every counter tracked by the caching allocator.

    The shape of the dictionary is documented under
    ``tensorflow_musa._C._device_allocator_stats`` and is covered by
    the stability test ``test_device_stats_schema_is_stable``. Keys
    include ``in_use_bytes``, ``reserved_bytes``, ``cache_hits`` /
    ``cache_misses``, ``oom_events``, per-segment counts, and the
    current ``limit_bytes`` cap. All values are non-negative ``int``.
    """
    return dict(_C._device_allocator_stats(_resolve_ordinal(device)))


def set_per_process_memory_fraction(
    fraction: float,
    device: Optional[int] = None,
) -> int:
    """Cap the allocator at ``fraction`` of the device's total memory.

    Requests that would push the driver-owned byte count above
    ``fraction * musaMemGetInfo.total`` return OOM immediately
    without touching the driver. Pass ``0.0`` to clear an existing
    cap. Values must satisfy ``0.0 < fraction <= 1.0``; any other
    value clears the limit (matching the C++ contract).

    Args:
        fraction: Fraction of the device's total memory, in (0, 1].
        device: Device ordinal (defaults to 0).

    Returns:
        The effective byte cap that was installed (0 when cleared).
    """
    ordinal = _resolve_ordinal(device)
    if not isinstance(fraction, (int, float)):
        raise TypeError(f"fraction must be numeric, got {type(fraction).__name__}")
    return int(_C._device_set_memory_fraction(float(fraction), ordinal))


def mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` from the MUSA driver.

    Thin wrapper over ``musaMemGetInfo`` — useful for dashboards that
    want raw driver numbers to cross-check the allocator's view of
    the world. Raises ``RuntimeError`` if the driver call fails.
    """
    free_b, total_b = _C._device_memory_usage(_resolve_ordinal(device))
    return int(free_b), int(total_b)


def get_allocator_backend() -> str:
    """Return the active device allocator backend name.

    ``"caching"`` when the MUSA caching allocator is active (the
    default and recommended mode), ``"passthrough"`` when
    ``TF_MUSA_DEVICE_ALLOCATOR=passthrough`` forces raw
    ``musaMalloc``/``musaFree`` for correctness bisection.

    Note: this is listed as "optional" in plan §6.3 but costs nearly
    nothing to expose since the underlying C helper is already
    there; ops dashboards rely on it to validate deployments.
    """
    return str(_C._device_allocator_backend())
