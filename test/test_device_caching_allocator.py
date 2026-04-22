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

"""End-to-end tests for the device-side caching allocator (plan commit C3).

These tests drive real traffic through the TF PluggableDevice bridge and
then read back the allocator's counters via the ``tensorflow_musa._C``
extension, which reaches the *same* singleton because it links against
``libmusa_core.so``. That makes this file a cross-library integration
test as well as a functional check.

The suite requires a working MUSA device. It skips (rather than fails)
when MUSA is not usable so CPU-only CI lanes remain green.
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


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run `script` in a fresh interpreter.

    TF 2.6 aborts when ``load_pluggable_device_library`` runs under the
    same process that has already been inspected by pytest's collection
    machinery; isolating the body into a subprocess keeps the check
    deterministic across pytest and bare-python invocations.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
    )


EXPECTED_DEVICE_STAT_KEYS = {
    "in_use_bytes",
    "reserved_bytes",
    "cached_bytes",
    "peak_in_use_bytes",
    "alloc_requests",
    "cache_hits",
    "cache_misses",
    "oom_events",
    "splits",
    "merges",
    "segments",
}


def test_device_stats_schema_is_stable():
    """Guard the public stats surface; a rename here is a Python-API break."""
    mod = _load_c_extension()
    stats = mod._device_allocator_stats(0)
    assert isinstance(stats, dict)
    assert set(stats.keys()) == EXPECTED_DEVICE_STAT_KEYS, stats
    for v in stats.values():
        assert isinstance(v, int) and v >= 0


def test_backend_string_is_known():
    mod = _load_c_extension()
    backend = mod._device_allocator_backend()
    assert backend in ("caching", "passthrough"), backend


_E2E_SCRIPT = textwrap.dedent(
    """
    import importlib.util, json, os, sys, sysconfig
    build = os.path.abspath("build")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    spec = importlib.util.spec_from_file_location(
        "tensorflow_musa._C", os.path.join(build, "_C" + ext))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod._device_allocator_backend() != "caching":
        print("SKIP backend=" + mod._device_allocator_backend())
        sys.exit(0)
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
    pre = mod._device_allocator_stats(0)
    with tf.device("/MUSA:0"):
        for _ in range(30):
            a = tf.random.uniform([256, 256], dtype=tf.float32)
            b = tf.random.uniform([256, 256], dtype=tf.float32)
            _ = tf.linalg.matmul(a, b).numpy()
    post = mod._device_allocator_stats(0)
    print("PRE=" + json.dumps(pre))
    print("POST=" + json.dumps(post))
    print("OK")
    """
)


def test_caching_backend_serves_most_allocations_from_cache():
    """End-to-end: a tight eager loop must reuse blocks under caching.

    Runs in a subprocess so pytest's collection doesn't interfere with
    ``load_pluggable_device_library`` (which aborts under some TF 2.6
    builds when invoked inside the already-warmed pytest interpreter).
    """
    os.environ["TF_MUSA_DEVICE_ALLOCATOR"] = "caching"
    proc = _run_in_subprocess(_E2E_SCRIPT)
    out = proc.stdout + proc.stderr
    if "SKIP" in proc.stdout.splitlines()[0:1] or any(
        line.startswith("SKIP") for line in proc.stdout.splitlines()
    ):
        pytest.skip(proc.stdout.strip())
    assert proc.returncode == 0, f"subprocess failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in out, out

    import json
    pre = post = None
    for line in proc.stdout.splitlines():
        if line.startswith("PRE="):
            pre = json.loads(line[4:])
        elif line.startswith("POST="):
            post = json.loads(line[5:])
    assert pre is not None and post is not None, out

    delta_req = post["alloc_requests"] - pre["alloc_requests"]
    delta_hit = post["cache_hits"] - pre["cache_hits"]
    assert delta_req > 0, f"allocator not exercised: {post}"
    hit_ratio = delta_hit / delta_req
    assert hit_ratio >= 0.90, (
        f"cache hit ratio {hit_ratio:.2%} below 90% threshold; "
        f"pre={pre} post={post}"
    )
    assert post["splits"] >= pre["splits"]
    assert post["merges"] >= pre["merges"]
    assert post["peak_in_use_bytes"] >= pre["peak_in_use_bytes"]
    assert post["segments"] >= 1


def test_empty_cache_returns_unsigned_count():
    mod = _load_c_extension()
    released = mod._device_empty_cache(0)
    assert isinstance(released, int) and released >= 0
