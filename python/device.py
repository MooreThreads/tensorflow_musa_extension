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

"""MUSA device-query helpers (plan §6.3).

These four entry points round out the minimal Python surface: just
enough to answer "is MUSA up?", "how many cards?", and "what am I
running on?". Anything richer (device properties, current device
setter, synchronize) is intentionally left to TF's own APIs
(``tf.config.list_physical_devices``, ``tf.test.is_gpu_available``)
— see plan §6.3 "明确不做" for the rationale.

Device discovery intentionally goes through TensorFlow rather than
through a direct MUSA runtime call. That matters because PluggableDevice
visibility can be constrained by env vars (``TF_CPP_MIN_LOG_LEVEL``,
``MUSA_VISIBLE_DEVICES``) that TF knows about but the MUSA runtime
does not; mirroring TF's view keeps the Python API honest about what
the framework will actually schedule on.
"""

from typing import List, Optional

from ._ext import _C

__all__ = [
    "device_count",
    "current_device",
    "get_device_name",
    "is_available",
    # Multi-device peer access (optional tier, plan §3.8).
    "can_access_peer",
    "enable_peer_access",
    "peer_access_snapshot",
]


def _list_musa_physical_devices():
    """Return the list of MUSA PhysicalDevice objects visible to TF.

    Imports tensorflow lazily so importing ``tensorflow_musa.device``
    before TF is available doesn't blow up. Returns an empty list on
    any failure (no-TF, plugin-not-loaded, no-hardware); callers
    treat an empty list as "no device" per the PyTorch contract.
    """
    try:
        import tensorflow as tf  # Lazy: avoid hard TF dep at import time.
    except ImportError:
        return []
    try:
        return [d for d in tf.config.list_physical_devices() if d.device_type == "MUSA"]
    except Exception:
        return []


def device_count() -> int:
    """Return the number of MUSA devices visible to TensorFlow.

    Zero means either the plugin isn't loaded, no MUSA device is
    attached, or ``MUSA_VISIBLE_DEVICES`` has hidden every card.
    """
    return len(_list_musa_physical_devices())


def current_device() -> int:
    """Return the default MUSA device ordinal (always 0 for now).

    TensorFlow's eager scheduler picks a device via ``tf.device(...)``
    scopes rather than a process-wide "current device" knob, so
    exposing a setter would be misleading. We surface the getter for
    parity with ``torch.musa.current_device`` and return 0 whenever
    at least one MUSA device is visible; callers expecting per-thread
    semantics should use ``tf.device`` instead.

    Returns:
        0 when a MUSA device is available.

    Raises:
        RuntimeError: when no MUSA device is visible, matching
            torch.musa's behavior on a CPU-only install.
    """
    if device_count() == 0:
        raise RuntimeError(
            "No MUSA device is available. Ensure the plugin is loaded "
            "(import tensorflow_musa) and a MUSA card is visible."
        )
    return 0


def get_device_name(device: Optional[int] = None) -> str:
    """Return a human-readable name for the given device ordinal.

    The name is taken from TF's PhysicalDevice description so it
    reflects whatever string the plugin registered (typically
    ``"MUSA"`` followed by an index; future plugin updates may
    enrich this with board / SM info without changing the API).

    Args:
        device: Device ordinal; ``None`` means "current device".

    Returns:
        The physical device name, or ``"MUSA:N"`` as a fallback when
        TF doesn't surface a descriptive name.

    Raises:
        IndexError: when ``device`` is out of range.
    """
    ordinal = 0 if device is None else int(device)
    devices = _list_musa_physical_devices()
    if ordinal < 0 or ordinal >= len(devices):
        raise IndexError(
            f"device {ordinal} out of range; {len(devices)} MUSA device(s) visible"
        )
    pd = devices[ordinal]
    return getattr(pd, "name", None) or f"MUSA:{ordinal}"


def is_available() -> bool:
    """Return ``True`` iff at least one MUSA device is visible to TF.

    Equivalent to ``device_count() > 0`` but makes call sites read
    better (``if tensorflow_musa.is_available(): ...``).
    """
    return device_count() > 0


# ---------------------------------------------------------------------------
# Peer-to-peer access (plan §3.8). Thin pass-throughs over the _C cache;
# documented here rather than in memory.py because they are about which
# *devices* can talk to each other, not about memory state.
# ---------------------------------------------------------------------------


def _check_ordinals(*values: int) -> None:
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"device ordinal must be int, got {type(v).__name__}"
            )
        if v < 0:
            raise ValueError(f"device ordinal must be >= 0, got {v}")


def can_access_peer(from_device: int, to_device: int) -> bool:
    """Return ``True`` iff ``from_device`` can DMA into ``to_device``.

    Result is cached after the first call per ordered pair, so this
    is cheap to call in a hot loop. Self-access (``from == to``) is
    always ``True``. Out-of-range ordinals return ``False`` rather
    than raising, matching the underlying MUSA runtime behavior.

    Asymmetric by definition: ``can_access_peer(0, 1)`` may differ
    from ``can_access_peer(1, 0)`` on topologies where only one
    direction is wired (e.g. asymmetric PCIe root complexes).
    """
    _check_ordinals(from_device, to_device)
    return bool(_C._peer_can_access(int(from_device), int(to_device)))


def enable_peer_access(from_device: int, to_device: int) -> bool:
    """Enable ``from_device → to_device`` peer access (idempotent).

    Returns ``True`` when access is live after the call (already-
    enabled pairs count as success). Returns ``False`` when the pair
    isn't supported by the hardware. Does not raise on already-enabled
    pairs — that's the common case when a training harness calls this
    in a preamble that may already have been run.

    Note: the plugin's built-in ``memcpy_dtod`` path does *not* yet
    auto-dispatch peer copies. Enabling peer access is still useful
    for user code that issues ``musaMemcpyPeerAsync`` directly (or
    that will, once the plugin picks up peer-aware dispatch in a
    follow-up commit).
    """
    _check_ordinals(from_device, to_device)
    return bool(_C._peer_enable_access(int(from_device), int(to_device)))


def peer_access_snapshot() -> List[dict]:
    """Return the cached peer-access table.

    Each entry is ``{"from": int, "to": int, "can_access": int,
    "enabled": bool}`` where ``can_access`` is -1 / 0 / 1 (unknown /
    no / yes). Only pairs that have been queried or enabled appear —
    the snapshot stays small on large clusters.
    """
    return [dict(e) for e in _C._peer_access_snapshot()]
