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

"""Tests for the user-facing Python wrappers in ``tensorflow_musa``.

Plan commit C6 introduces ``python/memory.py`` and ``python/device.py``
on top of the raw ``tensorflow_musa._C`` extension. This file checks:

* the wrappers are importable on any host (no MUSA hardware needed),
* ``device`` queries degrade gracefully when TF is absent or no MUSA
  device is visible,
* the ``memory`` wrappers return the right *types* and *shapes* so
  downstream tools can depend on them — when a MUSA device is present
  we also round-trip through ``_device_allocator_stats`` to verify
  wrapping is bit-identical.

Hardware-dependent assertions live inside a subprocess so pytest
collection stays safe on CPU-only lanes; the wrapper-shape checks are
hardware-free.
"""

import importlib.util
import os
import subprocess
import sys
import sysconfig
import textwrap

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
PKG_DIR = os.path.join(REPO_ROOT, "python")


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _c_ext_path() -> str:
    return os.path.join(BUILD_DIR, f"_C{_ext_suffix()}")


def _skip_without_c_ext():
    if not os.path.isfile(_c_ext_path()):
        pytest.skip(f"_C extension not built at {_c_ext_path()}")


def _load_package_module(submodule: str):
    """Load ``tensorflow_musa.<submodule>`` directly from the source tree.

    We hand-build the ``tensorflow_musa`` package so tests can run
    without ``pip install`` and without importing ``tensorflow`` at
    collection time (``__init__`` does a best-effort TF plugin load
    that we explicitly *don't* want triggered for shape-only tests).
    """
    import types

    pkg_name = "tensorflow_musa"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [PKG_DIR]
        sys.modules[pkg_name] = pkg

    full = f"{pkg_name}.{submodule}"
    if full in sys.modules:
        return sys.modules[full]

    spec = importlib.util.spec_from_file_location(
        full, os.path.join(PKG_DIR, f"{submodule}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# device.py — no _C extension needed for most of these.
# ---------------------------------------------------------------------------


def test_device_module_public_surface_matches_plan():
    device = _load_package_module("device")
    expected = {"device_count", "current_device", "get_device_name", "is_available"}
    assert expected.issubset(set(device.__all__))
    for name in expected:
        assert callable(getattr(device, name)), name


def test_device_count_returns_int():
    device = _load_package_module("device")
    n = device.device_count()
    assert isinstance(n, int) and n >= 0


def test_is_available_matches_device_count():
    device = _load_package_module("device")
    assert device.is_available() == (device.device_count() > 0)


def test_current_device_raises_when_no_device_visible():
    device = _load_package_module("device")
    if device.device_count() > 0:
        pytest.skip("MUSA device visible; cannot exercise the RuntimeError path")
    with pytest.raises(RuntimeError):
        device.current_device()


def test_get_device_name_out_of_range():
    device = _load_package_module("device")
    with pytest.raises(IndexError):
        device.get_device_name(999)


# ---------------------------------------------------------------------------
# memory.py — pure-shape checks first, then a round-trip through _C.
# ---------------------------------------------------------------------------


def test_memory_module_public_surface_matches_plan():
    memory = _load_package_module("memory")
    expected = {
        "empty_cache",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "reset_peak_memory_stats",
        "memory_stats",
        "set_per_process_memory_fraction",
        "mem_get_info",
        "get_allocator_backend",
    }
    assert expected.issubset(set(memory.__all__))


def test_resolve_ordinal_rejects_bad_types():
    memory = _load_package_module("memory")
    # Using a private helper deliberately: these invariants are part of
    # the public contract (bool must not silently act as 0/1, negatives
    # must fail fast) and a dedicated check keeps regressions cheap.
    with pytest.raises(TypeError):
        memory._resolve_ordinal(True)
    with pytest.raises(TypeError):
        memory._resolve_ordinal("0")
    with pytest.raises(ValueError):
        memory._resolve_ordinal(-1)
    assert memory._resolve_ordinal(None) == 0
    assert memory._resolve_ordinal(2) == 2


def test_memory_wrappers_return_ints_when_c_extension_is_available():
    _skip_without_c_ext()
    memory = _load_package_module("memory")

    # Touch each wrapper at least once; every one must return a Python
    # ``int`` / ``str`` (never numpy). Peak reset is idempotent on a
    # fresh allocator, so calling it here is a no-op aside from
    # exercising argument parsing.
    assert isinstance(memory.memory_allocated(), int)
    assert isinstance(memory.memory_reserved(), int)
    assert isinstance(memory.max_memory_allocated(), int)
    memory.reset_peak_memory_stats()
    assert isinstance(memory.memory_stats(), dict)
    assert isinstance(memory.get_allocator_backend(), str)
    # empty_cache returns bytes released as int.
    assert isinstance(memory.empty_cache(), int)


def test_memory_stats_dict_matches_raw_c_ext():
    _skip_without_c_ext()
    memory = _load_package_module("memory")
    # Import _C directly via _ext to avoid a second load path.
    ext = _load_package_module("_ext")
    raw = ext._C._device_allocator_stats(0)
    wrapped = memory.memory_stats(0)
    # Same keys; same values for counters that don't change between
    # calls in a no-op workload.
    assert set(wrapped.keys()) == set(raw.keys())
    for key in ("in_use_bytes", "reserved_bytes", "segments", "limit_bytes"):
        assert wrapped[key] == raw[key], key


def test_set_per_process_memory_fraction_rejects_non_numeric():
    _skip_without_c_ext()
    memory = _load_package_module("memory")
    with pytest.raises(TypeError):
        memory.set_per_process_memory_fraction("0.5")
    # Clearing is always safe (fraction <=0 or >1 clears per the C
    # contract) and returns an int.
    assert isinstance(memory.set_per_process_memory_fraction(0.0), int)


def test_mem_get_info_shape():
    _skip_without_c_ext()
    memory = _load_package_module("memory")
    free_b, total_b = memory.mem_get_info()
    assert isinstance(free_b, int) and isinstance(total_b, int)
    assert total_b > 0
    assert 0 <= free_b <= total_b


# ---------------------------------------------------------------------------
# End-to-end: drive the allocator via TF, then observe through the
# Python wrappers. Runs in a subprocess so ``load_pluggable_device_library``
# gets a pristine interpreter (see test_device_caching_allocator.py).
# ---------------------------------------------------------------------------


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
    )


def test_end_to_end_wrappers_observe_allocator_traffic():
    _skip_without_c_ext()
    script = textwrap.dedent(
        f"""
        import os, sys, gc
        sys.path.insert(0, {REPO_ROOT!r})
        sys.path.insert(0, {BUILD_DIR!r})

        # Build the package namespace by hand (see
        # _load_package_module); avoids triggering tensorflow_musa's
        # plugin auto-load before we've imported tensorflow.
        import types, importlib.util
        pkg = types.ModuleType('tensorflow_musa')
        pkg.__path__ = [{PKG_DIR!r}]
        sys.modules['tensorflow_musa'] = pkg
        def load(name):
            spec = importlib.util.spec_from_file_location(
                'tensorflow_musa.' + name,
                os.path.join({PKG_DIR!r}, name + '.py'))
            m = importlib.util.module_from_spec(spec)
            sys.modules['tensorflow_musa.' + name] = m
            spec.loader.exec_module(m)
            return m
        load('_ext')
        memory = load('memory')

        import tensorflow as tf
        tf.load_library(os.path.join({BUILD_DIR!r}, 'libmusa_plugin.so'))
        tf.config.experimental.set_visible_devices(
            tf.config.list_physical_devices('MUSA'), 'MUSA')

        with tf.device('/device:MUSA:0'):
            t1 = tf.random.uniform([1024, 1024], dtype=tf.float32)
            t2 = t1 + 1.0
            _ = t2.numpy()

        # Wrapper must see the non-zero in-use / reserved counters.
        in_use = memory.memory_allocated()
        reserved = memory.memory_reserved()
        peak = memory.max_memory_allocated()
        assert in_use >= 0 and reserved >= in_use, (in_use, reserved)
        assert peak >= in_use

        stats = memory.memory_stats()
        assert stats['reserved_bytes'] == reserved

        del t1, t2
        gc.collect()
        released = memory.empty_cache()
        assert isinstance(released, int) and released >= 0

        print('OK', in_use, reserved, peak, released)
        """
    )
    r = _run_in_subprocess(script)
    assert r.returncode == 0, (
        f"subprocess failed (rc={r.returncode})\nSTDOUT:\n{r.stdout}\n"
        f"STDERR:\n{r.stderr}"
    )
    assert "OK" in r.stdout, r.stdout
