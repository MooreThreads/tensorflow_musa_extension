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

"""Tests for the VMM expandable-segments path (plan commit C5).

Covers:
  * The driver-api probe returns sane answers on hosts with and
    without ``libmusa.so`` VMM support.
  * ``TF_MUSA_ALLOC_CONF`` parsing (defaults + the
    ``expandable_segments:true`` toggle).
  * End-to-end: when the env flag is set AND the device reports VMM
    support, real TF traffic goes through
    ``ExpandableSegment::Create`` and every live segment reports
    ``is_expandable=True``. When the flag is off (default), segments
    report ``is_expandable=False``.

The e2e case runs in a subprocess for the same reason
``test_device_caching_allocator`` does: ``load_pluggable_device_library``
aborts under TF 2.6 when invoked inside an interpreter pytest has
already warmed up.
"""

import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import textwrap

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.path.join(REPO_ROOT, "build")


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _load_c_extension():
    ext_path = os.path.join(BUILD_DIR, f"_C{_ext_suffix()}")
    if not os.path.isfile(ext_path):
        pytest.skip(f"_C extension not built at {ext_path}")
    spec = importlib.util.spec_from_file_location(
        "tensorflow_musa._C", ext_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_subprocess(script: str, env_extra: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


EXPECTED_CONFIG_KEYS = {
    "expandable_segments",
    "max_split_size_bytes",
    "roundup_power2_divisions",
    "garbage_collection_threshold",
    "raw",
}


def test_allocator_config_schema_is_stable():
    """The returned dict must always carry the full key set.

    Users parse this dict for dashboards and runbooks; a rename or
    drop is a Python API break.
    """
    mod = _load_c_extension()
    cfg = mod._allocator_config()
    assert isinstance(cfg, dict)
    assert set(cfg.keys()) == EXPECTED_CONFIG_KEYS, cfg
    assert isinstance(cfg["expandable_segments"], bool)
    assert isinstance(cfg["max_split_size_bytes"], int)
    assert isinstance(cfg["roundup_power2_divisions"], int)
    assert isinstance(cfg["garbage_collection_threshold"], float)
    assert isinstance(cfg["raw"], str)


def test_vmm_probes_return_booleans():
    """`_vmm_available` and `_vmm_supported` always return bool.

    Returning non-bool would break hosts that JSON-serialize the
    results; we've been bitten by that before on the host-allocator
    stats, so keep an explicit contract test.
    """
    mod = _load_c_extension()
    assert isinstance(mod._vmm_available(), bool)
    assert isinstance(mod._vmm_supported(0), bool)
    # granularity: 0 if unavailable, >0 if available. Always int.
    g = mod._vmm_granularity(0)
    assert isinstance(g, int) and g >= 0
    if mod._vmm_available() and mod._vmm_supported(0):
        assert g > 0, "VMM-supported device must report nonzero granularity"


def test_vmm_supported_implies_available():
    """If per-device probe says True, the process-wide probe must too.

    The per-device probe in driver_api.cc early-exits to False when
    the symbol table isn't complete, so this is a redundancy check on
    that invariant.
    """
    mod = _load_c_extension()
    if mod._vmm_supported(0):
        assert mod._vmm_available()


_E2E_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import importlib.util, json, os, sys, sysconfig
    build = os.path.abspath("build")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    spec = importlib.util.spec_from_file_location(
        "tensorflow_musa._C", os.path.join(build, "_C" + ext))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._allocator_config()
    print("CFG=" + json.dumps(cfg))
    print("VMM_AVAIL=" + str(mod._vmm_available()))
    # Force driver init via a runtime call so the capability probe
    # on this device gets an initialized driver to ask.
    try:
        mod._device_memory_usage(0)
    except Exception as e:
        print("SKIP no-device:" + repr(e)); sys.exit(0)
    print("VMM_SUPP=" + str(mod._vmm_supported(0)))
    if not (cfg["expandable_segments"] and mod._vmm_supported(0)):
        print("VMM_ACTIVE=False")
    else:
        print("VMM_ACTIVE=True")
    if mod._device_allocator_backend() != "caching":
        print("SKIP backend=" + mod._device_allocator_backend()); sys.exit(0)
    try:
        import tensorflow as tf
        from tensorflow.python.framework import load_library
    except Exception as e:
        print("SKIP tf-import:" + repr(e)); sys.exit(0)
    plugin = os.path.join(build, "libmusa_plugin.so")
    try:
        load_library.load_pluggable_device_library(plugin)
    except Exception as e:
        print("SKIP plugin-load:" + repr(e)); sys.exit(0)
    if not tf.config.list_physical_devices("MUSA"):
        print("SKIP no-musa"); sys.exit(0)
    with tf.device("/MUSA:0"):
        for _ in range(8):
            a = tf.random.uniform([256, 256], dtype=tf.float32)
            b = tf.random.uniform([256, 256], dtype=tf.float32)
            _ = tf.linalg.matmul(a, b).numpy()
    segs = mod._device_segment_snapshot(0)
    stats = mod._device_allocator_stats(0)
    print("SEGS=" + json.dumps(segs))
    print("STATS=" + json.dumps(stats))
    print("OK")
    """
)


def _parse_subprocess_output(stdout: str):
    out = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k] = v
    return out


def _run_e2e(env_extra):
    proc = _run_in_subprocess(_E2E_SCRIPT_TEMPLATE, env_extra=env_extra)
    for line in proc.stdout.splitlines():
        if line.startswith("SKIP"):
            pytest.skip(line)
    assert proc.returncode == 0, (
        f"subprocess failed:\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout, proc.stdout
    return _parse_subprocess_output(proc.stdout)


def test_expandable_segments_toggle_is_effective():
    """With the env flag ON and driver support, segments are VMM.

    This is the core acceptance test for C5. We run the eager workload
    twice — once with ``expandable_segments:true`` and once without —
    and assert the per-segment ``is_expandable`` flag tracks the
    config. Without this check the plumbing could silently no-op.
    """
    mod = _load_c_extension()
    if not mod._vmm_available():
        pytest.skip("libmusa.so does not export the VMM API on this host")

    # With the toggle ON. We also need a device that reports VMM
    # support; if it doesn't, the allocator will transparently fall
    # back to musaMalloc and segments would report is_expandable=False.
    # In that case the env-ON run is indistinguishable from OFF so we
    # just verify no crash and move on.
    on = _run_e2e({"TF_MUSA_ALLOC_CONF": "expandable_segments:true"})
    cfg_on = json.loads(on["CFG"])
    assert cfg_on["expandable_segments"] is True

    if on.get("VMM_ACTIVE") == "True":
        segs_on = json.loads(on["SEGS"])
        stats_on = json.loads(on["STATS"])
        assert stats_on["segments"] >= 1
        assert len(segs_on) == stats_on["segments"]
        assert all(s["is_expandable"] for s in segs_on), (
            f"expected every segment to be VMM-backed when the env "
            f"toggle + driver support are on; got {segs_on}"
        )

    # With the toggle OFF. This run should never produce expandable
    # segments regardless of device support.
    off = _run_e2e({"TF_MUSA_ALLOC_CONF": ""})
    cfg_off = json.loads(off["CFG"])
    assert cfg_off["expandable_segments"] is False
    segs_off = json.loads(off["SEGS"])
    stats_off = json.loads(off["STATS"])
    if stats_off["segments"] > 0:
        assert not any(s["is_expandable"] for s in segs_off), (
            f"expected no VMM segments when the env toggle is off, "
            f"got {segs_off}"
        )


def test_config_parses_kv_and_ignores_unknown_keys():
    """Parser accepts valid keys, tolerates unknowns and malformed pairs."""
    env = {
        "TF_MUSA_ALLOC_CONF": (
            "expandable_segments:True,"
            "max_split_size_mb:512,"
            "roundup_power2_divisions:4,"
            "garbage_collection_threshold:0.75,"
            "definitely_not_a_real_key:42"
        )
    }
    script = textwrap.dedent(
        """
        import importlib.util, json, os, sysconfig
        build = os.path.abspath("build")
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        spec = importlib.util.spec_from_file_location(
            "tensorflow_musa._C", os.path.join(build, "_C" + ext))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("CFG=" + json.dumps(mod._allocator_config()))
        """
    )
    proc = _run_in_subprocess(script, env_extra=env)
    assert proc.returncode == 0, proc.stderr
    cfg_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("CFG=")
    )
    cfg = json.loads(cfg_line[4:])
    assert cfg["expandable_segments"] is True
    assert cfg["max_split_size_bytes"] == 512 * 1024 * 1024
    assert cfg["roundup_power2_divisions"] == 4
    assert abs(cfg["garbage_collection_threshold"] - 0.75) < 1e-9
    # The unknown key should be mentioned on stderr but not affect cfg.
    assert "definitely_not_a_real_key" in proc.stderr, proc.stderr


def test_config_empty_env_gives_defaults():
    env = {"TF_MUSA_ALLOC_CONF": ""}
    script = textwrap.dedent(
        """
        import importlib.util, json, os, sysconfig
        build = os.path.abspath("build")
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        spec = importlib.util.spec_from_file_location(
            "tensorflow_musa._C", os.path.join(build, "_C" + ext))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("CFG=" + json.dumps(mod._allocator_config()))
        """
    )
    proc = _run_in_subprocess(script, env_extra=env)
    assert proc.returncode == 0, proc.stderr
    cfg_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("CFG=")
    )
    cfg = json.loads(cfg_line[4:])
    assert cfg == {
        "expandable_segments": False,
        "max_split_size_bytes": 0,
        "roundup_power2_divisions": 0,
        "garbage_collection_threshold": 0.0,
        "raw": "",
    }
