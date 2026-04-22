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

from typing import Optional

__all__ = [
    "device_count",
    "current_device",
    "get_device_name",
    "is_available",
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
