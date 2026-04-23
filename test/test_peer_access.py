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

"""Tests for the peer-to-peer access cache (plan §3.8, optional tier).

The interesting assertions need >= 2 MUSA devices, so those tests
skip gracefully on single-GPU boxes. The degenerate (self-access)
and argument-validation checks run everywhere.
"""

import importlib.util
import os
import sys
import sysconfig
import types

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
PKG_DIR = os.path.join(REPO_ROOT, "python")


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _skip_without_c_ext():
    if not os.path.isfile(os.path.join(BUILD_DIR, f"_C{_ext_suffix()}")):
        pytest.skip("_C extension not built")


def _load_submodule(name: str):
    pkg_name = "tensorflow_musa"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [PKG_DIR]
        sys.modules[pkg_name] = pkg
    full = f"{pkg_name}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(PKG_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


def _device_count() -> int:
    ext = _load_submodule("_ext")
    # Prefer the runtime's count — `device.device_count()` goes
    # through TF which requires importing tensorflow; here we want
    # to know how many devices the driver exposes so the test can
    # pick the right ordinals without standing up TF.
    return int(ext._C._peer_device_count())


# ---------------------------------------------------------------------------
# Argument validation & self-access (works on every host).
# ---------------------------------------------------------------------------


def test_can_access_peer_self_is_true():
    _skip_without_c_ext()
    if _device_count() == 0:
        pytest.skip("no MUSA device visible to the runtime")
    dev = _load_submodule("device")
    assert dev.can_access_peer(0, 0) is True


def test_can_access_peer_rejects_non_int():
    _skip_without_c_ext()
    dev = _load_submodule("device")
    with pytest.raises(TypeError):
        dev.can_access_peer(True, 1)
    with pytest.raises(TypeError):
        dev.can_access_peer("0", 1)
    with pytest.raises(ValueError):
        dev.can_access_peer(-1, 0)


def test_can_access_peer_out_of_range_returns_false():
    _skip_without_c_ext()
    dev = _load_submodule("device")
    # A ridiculous ordinal should return False, not throw — mirrors
    # the MUSA runtime's behavior and lets callers probe freely.
    assert dev.can_access_peer(0, 1_000_000) is False


def test_peer_access_snapshot_entries_are_dicts():
    _skip_without_c_ext()
    dev = _load_submodule("device")
    # Prime the cache so the snapshot isn't empty (if device count > 0).
    if _device_count() > 0:
        dev.can_access_peer(0, 0)
    snap = dev.peer_access_snapshot()
    assert isinstance(snap, list)
    for entry in snap:
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"from", "to", "can_access", "enabled"}


# ---------------------------------------------------------------------------
# Multi-device behavior (needs >= 2 MUSA devices).
# ---------------------------------------------------------------------------


def test_can_access_peer_between_different_devices():
    _skip_without_c_ext()
    n = _device_count()
    if n < 2:
        pytest.skip(f"need >= 2 MUSA devices, have {n}")
    dev = _load_submodule("device")
    # The result is hardware-specific; we only assert it's a bool.
    # What matters is the cached value is stable across repeated calls.
    a = dev.can_access_peer(0, 1)
    b = dev.can_access_peer(0, 1)
    assert isinstance(a, bool) and a == b


def test_enable_peer_access_is_idempotent():
    _skip_without_c_ext()
    n = _device_count()
    if n < 2:
        pytest.skip(f"need >= 2 MUSA devices, have {n}")
    dev = _load_submodule("device")
    if not dev.can_access_peer(0, 1):
        pytest.skip("device 0 cannot access device 1 (hardware topology)")
    assert dev.enable_peer_access(0, 1) is True
    # Second call must not raise and must still report success.
    assert dev.enable_peer_access(0, 1) is True
    # Snapshot should show the enabled entry.
    matching = [
        e for e in dev.peer_access_snapshot()
        if e["from"] == 0 and e["to"] == 1
    ]
    assert matching and matching[0]["enabled"] is True


def test_peer_device_count_matches_runtime():
    _skip_without_c_ext()
    ext = _load_submodule("_ext")
    n = int(ext._C._peer_device_count())
    assert isinstance(n, int) and n >= 0
