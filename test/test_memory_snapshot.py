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

"""Tests for ``tensorflow_musa.memory.memory_snapshot`` and its
companion ``_dump_snapshot`` / ``_record_memory_history`` helpers
(plan commit C6 / §5.6 snapshot tier).

The shape-level tests are hardware-free; the time-series and TF
round-trip tests skip gracefully when the _C extension or a MUSA
device isn't available.
"""

import importlib.util
import json
import os
import sys
import sysconfig
import time
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


# ---------------------------------------------------------------------------
# Shape / contract
# ---------------------------------------------------------------------------


def test_memory_snapshot_has_required_top_level_keys():
    _skip_without_c_ext()
    snap = _load_submodule("snapshot").memory_snapshot()
    required = {
        "device",
        "backend",
        "timestamp_ns",
        "stats",
        "segments",
        "driver",
        "allocator_config",
        "vmm",
        "last_oom_message",
    }
    missing = required - set(snap.keys())
    assert not missing, f"snapshot missing keys: {missing}"
    assert isinstance(snap["device"], int)
    assert isinstance(snap["backend"], str)
    assert isinstance(snap["stats"], dict)
    assert isinstance(snap["segments"], list)
    assert isinstance(snap["allocator_config"], dict)
    assert isinstance(snap["vmm"], dict)
    assert isinstance(snap["last_oom_message"], str)


def test_memory_snapshot_stats_match_direct_c_call():
    _skip_without_c_ext()
    snap_mod = _load_submodule("snapshot")
    ext = _load_submodule("_ext")
    raw = ext._C._device_allocator_stats(0)
    snap = snap_mod.memory_snapshot(0)
    # snapshot must carry *at least* the same keys; C-side and Python
    # views of the allocator share one singleton so values must line up
    # for stable counters.
    for key in ("reserved_bytes", "segments", "limit_bytes"):
        assert snap["stats"][key] == raw[key], key


def test_memory_snapshot_driver_section_ok_or_error():
    _skip_without_c_ext()
    snap = _load_submodule("snapshot").memory_snapshot()
    driver = snap["driver"]
    if "error" in driver:
        assert isinstance(driver["error"], str)
    else:
        assert set(driver.keys()) >= {"free_bytes", "total_bytes"}
        assert 0 <= driver["free_bytes"] <= driver["total_bytes"]


# ---------------------------------------------------------------------------
# _dump_snapshot — atomic JSON write
# ---------------------------------------------------------------------------


def test_dump_snapshot_round_trips_to_json(tmp_path):
    _skip_without_c_ext()
    snap_mod = _load_submodule("snapshot")
    target = tmp_path / "snap.json"
    written = snap_mod._dump_snapshot(str(target))
    assert os.path.isfile(written)
    with open(written, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # The file's shape must match an in-memory snapshot (modulo the
    # timestamp, which advances).
    for key in ("device", "backend", "stats", "segments", "allocator_config"):
        assert key in payload


def test_dump_snapshot_cleans_up_tmp_on_success(tmp_path):
    _skip_without_c_ext()
    snap_mod = _load_submodule("snapshot")
    target = tmp_path / "snap.json"
    snap_mod._dump_snapshot(str(target))
    # .tmp sibling must not linger after a successful write.
    assert not (tmp_path / "snap.json.tmp").exists()


# ---------------------------------------------------------------------------
# _record_memory_history — polling sampler
# ---------------------------------------------------------------------------


def test_record_memory_history_samples_and_stops():
    _skip_without_c_ext()
    snap_mod = _load_submodule("snapshot")
    snap_mod._record_memory_history(True, max_entries=64, interval_ms=5)
    try:
        # Give the sampler a handful of ticks.
        time.sleep(0.08)
        snap = snap_mod.memory_snapshot()
    finally:
        snap_mod._record_memory_history(False)

    assert "history" in snap, snap.keys()
    assert snap["history"]["active"] is True
    entries = snap["history"]["entries"]
    assert len(entries) >= 1
    # Each entry must carry the core stats columns plus a monotonic
    # time-offset in nanoseconds.
    e0 = entries[0]
    for key in ("in_use_bytes", "reserved_bytes", "t_ns"):
        assert key in e0, key
    t_values = [e["t_ns"] for e in entries]
    assert t_values == sorted(t_values), "time offsets must be monotonic"


def test_record_memory_history_respects_max_entries():
    _skip_without_c_ext()
    snap_mod = _load_submodule("snapshot")
    snap_mod._record_memory_history(True, max_entries=4, interval_ms=2)
    try:
        time.sleep(0.15)  # ~75 ticks at 2ms, ring-buffer should trim.
    finally:
        snap_mod._record_memory_history(False)
    entries = snap_mod.memory_snapshot()["history"]["entries"]
    assert len(entries) <= 4


def test_record_memory_history_disabled_by_default():
    snap_mod = _load_submodule("snapshot")
    # No call to _record_memory_history → history is either absent
    # (never recorded) or present with active=False and whatever
    # entries remain from a prior test. We just check the module
    # doesn't spin up a thread on its own.
    assert not snap_mod._recorder.is_active()


def test_record_memory_history_rejects_bad_device_type():
    snap_mod = _load_submodule("snapshot")
    with pytest.raises(TypeError):
        snap_mod._record_memory_history(True, device="0")


# ---------------------------------------------------------------------------
# memory.py re-exports
# ---------------------------------------------------------------------------


def test_memory_reexports_snapshot_helpers():
    _skip_without_c_ext()
    memory = _load_submodule("memory")
    for name in ("memory_snapshot", "_dump_snapshot", "_record_memory_history"):
        assert hasattr(memory, name), name
    assert "memory_snapshot" in memory.__all__
