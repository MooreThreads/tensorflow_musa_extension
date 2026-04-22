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

"""Layout check for the split-build artifacts (plan commit C2).

These checks run against the build directory; they verify that the three
cooperating shared objects were produced and that their RPATH / NEEDED
wiring is sound. A follow-up wheel-install check will run under the
eventual CI matrix; for now this file documents the required layout so
regressions are caught at commit time.
"""

import importlib.util
import os
import subprocess
import sysconfig

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUILD_DIR = os.path.join(REPO_ROOT, "build")


def _ext_suffix() -> str:
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    return ext if ext else ".so"


EXPECTED_ARTIFACTS = (
    "libmusa_core.so",
    "libmusa_plugin.so",
    f"_C{_ext_suffix()}",
)


def _readelf_dynamic(path: str) -> str:
    return subprocess.check_output(
        ["readelf", "-d", path], stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace")


def test_all_artifacts_exist():
    missing = [
        name
        for name in EXPECTED_ARTIFACTS
        if not os.path.isfile(os.path.join(BUILD_DIR, name))
    ]
    assert not missing, (
        f"missing split-build artifacts in {BUILD_DIR}: {missing}. "
        "Run `cmake .. && make` first."
    )


@pytest.mark.parametrize(
    "consumer",
    ["libmusa_plugin.so", f"_C{_ext_suffix()}"],
)
def test_consumers_link_against_core(consumer):
    path = os.path.join(BUILD_DIR, consumer)
    if not os.path.isfile(path):
        pytest.skip(f"{consumer} not built")
    dyn = _readelf_dynamic(path)
    assert "libmusa_core.so" in dyn, (
        f"{consumer} must depend on libmusa_core.so so both consumers share a "
        f"singleton:\n{dyn}"
    )


def test_core_has_origin_rpath():
    dyn = _readelf_dynamic(os.path.join(BUILD_DIR, "libmusa_core.so"))
    assert "$ORIGIN" in dyn, (
        "libmusa_core.so RPATH must contain $ORIGIN so the plugin and _C can "
        f"locate it when installed alongside them:\n{dyn}"
    )


def test_c_extension_is_importable():
    ext_path = os.path.join(BUILD_DIR, f"_C{_ext_suffix()}")
    if not os.path.isfile(ext_path):
        pytest.skip("_C extension not built")
    spec = importlib.util.spec_from_file_location("tensorflow_musa._C", ext_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._is_loaded() is True
    stats = mod._host_allocator_stats()
    assert isinstance(stats, dict)
    # Exact keys must match the HostCachingAllocatorStats struct.
    expected = {
        "in_use_bytes",
        "cached_bytes",
        "total_reserved_bytes",
        "peak_bytes",
        "alloc_requests",
        "cache_hits",
        "cache_misses",
        "pool_cap_rejections",
        "record_stream_count",
    }
    assert set(stats.keys()) == expected, stats
